import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import telebot
from telebot import types

AMPUTATION_PERCENTAGES = {
    'male': {
        'shoulder': 3.25, 'forearm': 1.87, 'hand': 0.65,
        'thigh': 10.50, 'calf': 4.75, 'foot': 1.43
    },
    'female': {
        'shoulder': 2.90, 'forearm': 1.57, 'hand': 0.50,
        'thigh': 11.80, 'calf': 5.35, 'foot': 1.33
    }
}

def prepare_amputation_dataset_v2(
    df: pd.DataFrame,
    side_available: str = "R",
    target_mode: str = "fallback",
    use_weight_corrected: bool = True,
    limb_mass_col: str = "Estimated_limb_mass_lost_kg",
    amputation_level_col: str = "Amputation_level",
    days_after_col: str = "Days_after_amputation",
    dropna: bool = True
):
    df = df.copy()
    s = side_available.upper().strip()

    needed = [
        "Whole_grain_products", "Age",
        "ThicknessSkinFatParaumbilicalAreaRectusAbdominisMuscle",
        "ThicknessSkinFatParaumbilicalAreaRectusAbdominisMuscle_Re_Examination",
        "Body_Weight", "Body_Weight_Re_Examination",
        "WaistCircumference", "WaistCircumference_Re_Examination",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    shoulder_0 = f"Shoulder_{s}"
    shoulder_1 = f"Shoulder_{s}_Re_Examination"
    grip_0 = f"Dynamometry_{s}"
    grip_1 = f"Dynamometry_{s}_Re_Examination"
    for c in [shoulder_0, shoulder_1, grip_0, grip_1]:
        if c not in df.columns:
            df[c] = np.nan

    df["Delta_Shoulder"] = df[shoulder_1] - df[shoulder_0]
    df["Delta_Grip"] = df[grip_1] - df[grip_0]

    df["Delta_Skinfold"] = (
        df["ThicknessSkinFatParaumbilicalAreaRectusAbdominisMuscle_Re_Examination"]
        - df["ThicknessSkinFatParaumbilicalAreaRectusAbdominisMuscle"]
    )
    df["Delta_Weight"] = df["Body_Weight_Re_Examination"] - df["Body_Weight"]
    df["Delta_Waist"] = df["WaistCircumference_Re_Examination"] - df["WaistCircumference"]

    if amputation_level_col not in df.columns:
        df[amputation_level_col] = "unknown"
    if days_after_col not in df.columns:
        df[days_after_col] = np.nan
    if limb_mass_col not in df.columns:
        df[limb_mass_col] = np.nan

    df["BodyWeight_corrected"] = df["Body_Weight"]
    df["BodyWeight_corrected_Re_Exam"] = df["Body_Weight_Re_Examination"]

    if use_weight_corrected:
        known = df[limb_mass_col].notna()
        df.loc[known, "BodyWeight_corrected"] = df.loc[known, "Body_Weight"] + df.loc[known, limb_mass_col]
        df.loc[known, "BodyWeight_corrected_Re_Exam"] = (
            df.loc[known, "Body_Weight_Re_Examination"] + df.loc[known, limb_mass_col]
        )
        df["Delta_Weight_corrected"] = df["BodyWeight_corrected_Re_Exam"] - df["BodyWeight_corrected"]
    else:
        df["Delta_Weight_corrected"] = df["Delta_Weight"]

    if target_mode == "upper":
        df["Muscle_mass_proxy"] = df[["Delta_Shoulder", "Delta_Grip"]].sum(axis=1, min_count=1)
    elif target_mode == "fallback":
        df["Muscle_mass_proxy"] = df[["Delta_Grip", "Delta_Shoulder"]].sum(axis=1, min_count=1)
    else:
        raise ValueError("target_mode must be 'upper' or 'fallback'")

    feat_cols = [
        "Whole_grain_products",
        "Age",
        "Delta_Skinfold",
        "Delta_Weight_corrected",
        "Delta_Waist",
        days_after_col,
        limb_mass_col,
    ]

    level_dummies = pd.get_dummies(df[amputation_level_col].astype(str).str.lower(), prefix="amp_level")
    df = pd.concat([df, level_dummies], axis=1)
    feat_cols += list(level_dummies.columns)

    X = df[feat_cols]
    y = df["Muscle_mass_proxy"]

    if dropna:
        must_have = ["Whole_grain_products", "Age", "Delta_Skinfold", "Delta_Weight_corrected", "Delta_Waist", "Muscle_mass_proxy"]
        data = pd.concat([X, y], axis=1).dropna(subset=must_have)
        X = data[feat_cols]
        y = data["Muscle_mass_proxy"]

    return X, y, feat_cols

def build_new_person_features_v2(new_person: dict, feat_cols: list):
    row = dict(new_person)
    row.setdefault("Days_after_amputation", np.nan)
    row.setdefault("Estimated_limb_mass_lost_kg", np.nan)
    row.setdefault("Amputation_level", "unknown")

    if "Delta_Weight_corrected" not in row:
        if all(k in row for k in ["Body_Weight", "Body_Weight_Re_Examination"]):
            bw0 = row["Body_Weight"]
            bw1 = row["Body_Weight_Re_Examination"]
            lost = row.get("Estimated_limb_mass_lost_kg", np.nan)
            if pd.notna(lost):
                row["Delta_Weight_corrected"] = (bw1 + lost) - (bw0 + lost)
            else:
                row["Delta_Weight_corrected"] = bw1 - bw0
        else:
            row["Delta_Weight_corrected"] = np.nan

    level = str(row.get("Amputation_level", "unknown")).lower()
    for c in feat_cols:
        if c.startswith("amp_level_"):
            row[c] = 1.0 if c == f"amp_level_{level}" else 0.0

    new_df = pd.DataFrame([row])
    for c in feat_cols:
        if c not in new_df.columns:
            new_df[c] = np.nan

    return new_df[feat_cols]

def Gastrointestinal_Tract_Symptoms(Whole_grain_products, Age, Height, Body_Weight, Body_Weight_Re_Examination, Heartburn):
    try:
        df = pd.read_csv("Rehabilitation_imputed_whole_grain.csv")
    except FileNotFoundError:
        return {}

    height_m = Height

    bmi_old = Body_Weight / (height_m ** 2)
    bmi_new = Body_Weight_Re_Examination / (height_m ** 2)
    bmi_delta = bmi_new - bmi_old

    features = ["Whole_grain_products", "Age", "BMI_Delta"]

    new_person = pd.DataFrame([{
        "Whole_grain_products": Whole_grain_products,
        "Age": Age,
        "BMI_Delta": bmi_delta,
    }], columns=features)

    targets = {
        'Heartburn_Re_Examination': 'Heartburn',
        'Constipation_Re_Examination': 'Constipation',
        'Bloating_Re_Examination': 'Bloating',
        'Insomnia_Re_Examination': 'Insomnia'
    }
    
    cols_to_clean = ["Whole_grain_products", "Age", "Height", "Body_Weight", "Body_Weight_Re_Examination", "Heartburn"] + list(targets.keys())
    for col in cols_to_clean:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    df["Height_m"] = df["Height"] / 100
    df["BMI_Delta"] = (df["Body_Weight_Re_Examination"] / (df["Height_m"] ** 2)) - \
                      (df["Body_Weight"] / (df["Height_m"] ** 2))

    results_dict = {}

    for col_name, label in targets.items():
        if col_name in df.columns:

            temp_df = df.dropna(subset=features + [col_name]).copy()
            temp_df["Symptom_gone"] = (temp_df[col_name] == 0).astype(int)

            X = temp_df[features]
            y = temp_df["Symptom_gone"]
            
            if len(y.unique()) < 2:
                continue
                
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.25,
                random_state=42
            )
            
            model_rf = RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced"
            )
            model_rf.fit(X_train, y_train)
            
            model_lr = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs"
            )
            model_lr.fit(X_train, y_train)
            
            probability_success = model_lr.predict_proba(new_person)[0][1]

            importances_rf = pd.Series(model_rf.feature_importances_, index=features)
            imp_dict_rf = importances_rf.sort_values(ascending=False).round(4).to_dict()

            coef_lr = pd.Series(model_lr.coef_[0], index=features)
            coef_dict_lr = coef_lr.sort_values(ascending=False).round(4).to_dict()
            
            results_dict[label] = {
                "prob": probability_success,
                "imp_rf": imp_dict_rf,
                "coef_lr": coef_dict_lr
            }

    return results_dict

def Predict_Muscle_Mass_Primary(Whole_grain_products, Age):
    try:
        df = pd.read_csv("Rehabilitation_imputed_whole_grain.csv")
    except FileNotFoundError:
        return None

    num_cols = ["Whole_grain_products", "Age", "Shoulder_R", "Shoulder_R_Re_Examination", "Dynamometry_R", "Dynamometry_R_Re_Examination"]
    for col in num_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if not all(c in df.columns for c in ["Shoulder_R", "Shoulder_R_Re_Examination", "Dynamometry_R", "Dynamometry_R_Re_Examination"]):
        return None

    df["Delta_Shoulder"] = df["Shoulder_R_Re_Examination"] - df["Shoulder_R"]
    df["Delta_Grip"] = df["Dynamometry_R_Re_Examination"] - df["Dynamometry_R"]
    df["Muscle_mass"] = df["Delta_Shoulder"] + df["Delta_Grip"]

    X = df[["Whole_grain_products", "Age"]]
    y = df["Muscle_mass"]

    data = pd.concat([X, y], axis=1).dropna()
    if data.empty:
        return None

    X = data[["Whole_grain_products", "Age"]]
    y = data["Muscle_mass"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = RandomForestRegressor(n_estimators=400, random_state=42)
    model.fit(X_train, y_train)

    new_person = pd.DataFrame([{"Whole_grain_products": Whole_grain_products, "Age": Age}])
    return model.predict(new_person)[0]

def Predict_Muscle_Mass_Secondary(Whole_grain_products, Age, Delta_Weight, Delta_Waist, Delta_Skinfold):
    try:
        df = pd.read_csv("Rehabilitation_imputed_whole_grain.csv")
    except FileNotFoundError:
        return None

    num_cols = [
        "Whole_grain_products", "Age", "Shoulder_R", "Shoulder_R_Re_Examination",
        "Dynamometry_R", "Dynamometry_R_Re_Examination",
        "ThicknessSkinFatParaumbilicalAreaRectusAbdominisMuscle",
        "ThicknessSkinFatParaumbilicalAreaRectusAbdominisMuscle_Re_Examination",
        "Body_Weight", "Body_Weight_Re_Examination",
        "WaistCircumference", "WaistCircumference_Re_Examination"
    ]

    for col in num_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_cols = ["Shoulder_R", "Shoulder_R_Re_Examination", "Dynamometry_R", "Dynamometry_R_Re_Examination",
                     "ThicknessSkinFatParaumbilicalAreaRectusAbdominisMuscle_Re_Examination", "ThicknessSkinFatParaumbilicalAreaRectusAbdominisMuscle",
                     "WaistCircumference_Re_Examination", "WaistCircumference", "Body_Weight_Re_Examination", "Body_Weight"]

    if not all(c in df.columns for c in required_cols):
        return None

    df["Delta_Shoulder"] = df["Shoulder_R_Re_Examination"] - df["Shoulder_R"]
    df["Delta_Grip"] = df["Dynamometry_R_Re_Examination"] - df["Dynamometry_R"]
    df["Muscle_mass"] = df["Delta_Shoulder"] + df["Delta_Grip"]

    df["Delta_Skinfold"] = df["ThicknessSkinFatParaumbilicalAreaRectusAbdominisMuscle_Re_Examination"] - df["ThicknessSkinFatParaumbilicalAreaRectusAbdominisMuscle"]
    df["Delta_Weight"] = df["Body_Weight_Re_Examination"] - df["Body_Weight"]
    df["Delta_Waist"] = df["WaistCircumference_Re_Examination"] - df["WaistCircumference"]

    features = ["Whole_grain_products", "Age", "Delta_Skinfold", "Delta_Weight", "Delta_Waist"]
    data = df[features + ["Muscle_mass"]].dropna()

    if data.empty:
        return None

    X = data[features]
    y = data["Muscle_mass"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)

    new_person = pd.DataFrame([{
        "Whole_grain_products": Whole_grain_products,
        "Age": Age,
        "Delta_Skinfold": Delta_Skinfold,
        "Delta_Weight": Delta_Weight,
        "Delta_Waist": Delta_Waist
    }])

    return model.predict(new_person)[0]

bot = telebot.TeleBot('8520830685:AAGvGEkMvKkecglIwAcfgVORvGYlq7Vd81w')
patient_symptoms = {}

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.clear_step_handler_by_chat_id(message.chat.id)
    patient_symptoms[message.chat.id] = {'snaq_score': 0}

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    markup.add("Male", "Female")

    msg = bot.send_message(
        message.chat.id,
        "Hello! 👋 \nI am SmartRecover AI bot for muscle strength and motor coordination recovery. I analyze progress and adapt loads to your current condition. I offer personalized treatment plans and nutritional advice.\n\nPlease select your gender:",
        reply_markup=markup
    )
    bot.register_next_step_handler(msg, get_gender)

def get_gender(message):
    if message.text == "Male":
        patient_symptoms[message.chat.id]['Gender'] = 'male'
    elif message.text == "Female":
        patient_symptoms[message.chat.id]['Gender'] = 'female'
    else:
        patient_symptoms[message.chat.id]['Gender'] = 'male'

    markup = types.ReplyKeyboardRemove()
    msg = bot.send_message(message.chat.id, "Enter your age (full years):", reply_markup=markup)
    bot.register_next_step_handler(msg, get_age)

def get_age(message):
    if message.text == '/start':
        start_message(message)
        return
    try:
        patient_symptoms[message.chat.id]['Age'] = int(message.text)
        start_snaq_question_1(message.chat.id)
    except ValueError:
        msg = bot.send_message(message.chat.id, "Please enter a valid integer for age.")
        bot.register_next_step_handler(msg, get_age)

def start_snaq_question_1(chat_id):
    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("Very bad", callback_data="snaq1_1"))
    markup.add(types.InlineKeyboardButton("Bad", callback_data="snaq1_2"))
    markup.add(types.InlineKeyboardButton("Average", callback_data="snaq1_3"))
    markup.add(types.InlineKeyboardButton("Good", callback_data="snaq1_4"))
    markup.add(types.InlineKeyboardButton("Very good", callback_data="snaq1_5"))

    bot.send_message(chat_id, "SNAQ Questionnaire (1/4): Rate your appetite:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith("snaq1_"))
def handle_snaq_1(call):
    chat_id = call.message.chat.id
    score = int(call.data.split('_')[1])
    patient_symptoms[chat_id]['snaq_score'] += score

    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("Very tasteless", callback_data="snaq2_1"))
    markup.add(types.InlineKeyboardButton("Tasteless", callback_data="snaq2_2"))
    markup.add(types.InlineKeyboardButton("Average", callback_data="snaq2_3"))
    markup.add(types.InlineKeyboardButton("Tasty", callback_data="snaq2_4"))
    markup.add(types.InlineKeyboardButton("Very tasty", callback_data="snaq2_5"))

    bot.edit_message_text("SNAQ Questionnaire (2/4): How does the food taste to you?", chat_id, call.message.message_id, reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith("snaq2_"))
def handle_snaq_2(call):
    chat_id = call.message.chat.id
    score = int(call.data.split('_')[1])
    patient_symptoms[chat_id]['snaq_score'] += score

    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("Full after a few spoons", callback_data="snaq3_1"))
    markup.add(types.InlineKeyboardButton("Full after 1/3 of the meal", callback_data="snaq3_2"))
    markup.add(types.InlineKeyboardButton("Full after 1/2 of the meal", callback_data="snaq3_3"))
    markup.add(types.InlineKeyboardButton("Full after the whole meal", callback_data="snaq3_4"))
    markup.add(types.InlineKeyboardButton("Rarely eat to my heart's content", callback_data="snaq3_5"))

    bot.edit_message_text("SNAQ Questionnaire (3/4): When I eat, I feel...", chat_id, call.message.message_id, reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith("snaq3_"))
def handle_snaq_3(call):
    chat_id = call.message.chat.id
    score = int(call.data.split('_')[1])
    patient_symptoms[chat_id]['snaq_score'] += score

    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("Less than 1 meal", callback_data="snaq4_1"))
    markup.add(types.InlineKeyboardButton("1 meal", callback_data="snaq4_2"))
    markup.add(types.InlineKeyboardButton("2 meals", callback_data="snaq4_3"))
    markup.add(types.InlineKeyboardButton("3 meals", callback_data="snaq4_4"))
    markup.add(types.InlineKeyboardButton("More than 3 meals", callback_data="snaq4_5"))

    bot.edit_message_text("SNAQ Questionnaire (4/4): Usually, in a day I have...", chat_id, call.message.message_id, reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith("snaq4_"))
def handle_snaq_4(call):
    chat_id = call.message.chat.id
    score = int(call.data.split('_')[1])
    patient_symptoms[chat_id]['snaq_score'] += score

    total_score = patient_symptoms[chat_id]['snaq_score']
    if total_score <= 14:
        bot.send_message(chat_id, f"⚠️ *ATTENTION! Your SNAQ score: {total_score}*. \nThis indicates a significant risk of weight loss and malnutrition. Consultation with a dietitian is recommended.", parse_mode="Markdown")
    else:
        bot.send_message(chat_id, f"✅ *Your SNAQ score: {total_score}*. \nRisk of malnutrition is low.", parse_mode="Markdown")

    msg = bot.send_message(chat_id, "Enter the amount of whole grain products (grams per day):")
    bot.register_next_step_handler(msg, get_grains)

def get_grains(message):
    try:
        val = float(message.text.replace(',', '.'))
        patient_symptoms[message.chat.id]['Whole_grain_products'] = val

        msg = bot.send_message(
            message.chat.id,
            "Enter your height (in meters, e.g., 1.75).\n\n"
            "❗️ *If your height cannot be measured due to amputation*, enter `0`. "
            "The bot will calculate your estimated height based on knee height.",
            parse_mode="Markdown"
        )
        bot.register_next_step_handler(msg, get_height)
    except ValueError:
        msg = bot.send_message(message.chat.id, "Please enter a valid number.")
        bot.register_next_step_handler(msg, get_grains)

def get_height(message):
    chat_id = message.chat.id
    text = message.text.strip().replace(',', '.')

    if text == '0':
        msg = bot.send_message(
            chat_id,
            "For alternative height calculation, measure your *Knee Height (KH)*:\n"
            "Bend your knee at a 90° angle and measure the distance from the heel to the anterior surface of the thigh (above the kneecap).\n\n"
            "Enter the obtained value in *centimeters* (e.g., 52):",
            parse_mode="Markdown"
        )
        bot.register_next_step_handler(msg, calculate_alternative_height)
        return

    try:
        val = float(text)
        if val > 53:
            val = val / 100
        patient_symptoms[chat_id]['Height'] = val
        msg = bot.send_message(chat_id, "Thank you! Enter your INITIAL weight (before treatment, kg):")
        bot.register_next_step_handler(msg, get_weight_start)
    except ValueError:
        msg = bot.send_message(chat_id, "Please enter a valid number.")
        bot.register_next_step_handler(msg, get_height)

def calculate_alternative_height(message):
    chat_id = message.chat.id
    try:
        vk = float(message.text.replace(',', '.'))
        gender = patient_symptoms[chat_id].get('Gender', 'male')
        age = patient_symptoms[chat_id].get('Age', 30)

        if gender == 'male':
            height_cm = (2.02 * vk) - (0.04 * age) + 64.19
        else:
            height_cm = (1.83 * vk) - (0.24 * age) + 84.88

        height_m = height_cm / 100.0
        patient_symptoms[chat_id]['Height'] = height_m

        bot.send_message(chat_id, f"✅ Estimated calculated height: *{round(height_m, 2)} m*", parse_mode="Markdown")

        msg = bot.send_message(chat_id, "Thank you! Enter your INITIAL weight (before treatment, kg):")
        bot.register_next_step_handler(msg, get_weight_start)
    except ValueError:
        msg = bot.send_message(chat_id, "Please enter a valid number in centimeters.")
        bot.register_next_step_handler(msg, calculate_alternative_height)

def get_weight_start(message):
    try:
        val = float(message.text.replace(',', '.'))
        if val <= 0: raise ValueError
        patient_symptoms[message.chat.id]['Body_Weight'] = val
        msg = bot.send_message(message.chat.id, "Enter your weight AFTER rehabilitation (kg):")
        bot.register_next_step_handler(msg, get_weight_final)
    except ValueError:
        bot.send_message(message.chat.id, "Please enter a valid positive number.")
        bot.register_next_step_handler(message, get_weight_start)

def get_weight_final(message):
    try:
        val = float(message.text.replace(',', '.'))
        if val <= 0: raise ValueError
        patient_symptoms[message.chat.id]['Body_Weight_Re_Examination'] = val
        msg = bot.send_message(message.chat.id, "Enter your INITIAL waist circumference before treatment (cm):")
        bot.register_next_step_handler(msg, get_waist_start)
    except ValueError:
        bot.send_message(message.chat.id, "Please enter a valid number.")
        bot.register_next_step_handler(message, get_weight_final)

def get_waist_start(message):
    try:
        val = float(message.text.replace(',', '.'))
        patient_symptoms[message.chat.id]['Waist_Start'] = val
        msg = bot.send_message(message.chat.id, "Enter your waist circumference AFTER rehabilitation (cm):")
        bot.register_next_step_handler(msg, get_waist_final)
    except ValueError:
        bot.send_message(message.chat.id, "Please enter a valid number.")
        bot.register_next_step_handler(message, get_waist_start)

def get_waist_final(message):
    try:
        val = float(message.text.replace(',', '.'))
        patient_symptoms[message.chat.id]['Waist_Final'] = val
        msg = bot.send_message(message.chat.id, "Enter your INITIAL skinfold thickness (mm):")
        bot.register_next_step_handler(msg, get_skinfat_start)
    except ValueError:
        bot.send_message(message.chat.id, "Please enter a valid number.")
        bot.register_next_step_handler(message, get_waist_final)

def get_skinfat_start(message):
    try:
        val = float(message.text.replace(',', '.'))
        patient_symptoms[message.chat.id]['SkinFat_Start'] = val
        msg = bot.send_message(message.chat.id, "Enter your skinfold thickness AFTER rehabilitation (mm):")
        bot.register_next_step_handler(msg, get_skinfat_final)
    except ValueError:
        bot.send_message(message.chat.id, "Please enter a valid number.")
        bot.register_next_step_handler(message, get_skinfat_start)

def get_skinfat_final(message):
    try:
        val = float(message.text.replace(',', '.'))
        patient_symptoms[message.chat.id]['SkinFat_Final'] = val

        msg = bot.send_message(message.chat.id, "Enter the INITIAL circumference of the right shoulder (cm):")
        bot.register_next_step_handler(msg, get_shoulder_start)
    except ValueError:
        bot.send_message(message.chat.id, "Please enter a valid number.")
        bot.register_next_step_handler(message, get_skinfat_final)

def get_shoulder_start(message):
    try:
        val = float(message.text.replace(',', '.'))
        patient_symptoms[message.chat.id]['Shoulder_R_Start'] = val

        msg = bot.send_message(message.chat.id, "Enter the circumference of the right shoulder AFTER rehabilitation (cm):")
        bot.register_next_step_handler(msg, get_shoulder_final)
    except ValueError:
        bot.send_message(message.chat.id, "Please enter a valid number.")
        bot.register_next_step_handler(message, get_shoulder_start)

def get_shoulder_final(message):
    try:
        val = float(message.text.replace(',', '.'))
        patient_symptoms[message.chat.id]['Shoulder_R_Final'] = val
        perform_prediction(message.chat.id)
    except ValueError:
        bot.send_message(message.chat.id, "Please enter a valid number.")
        bot.register_next_step_handler(message, get_shoulder_final)

def perform_prediction(chat_id):
    try:
        data = patient_symptoms[chat_id]

        all_results = Gastrointestinal_Tract_Symptoms(
            Whole_grain_products=data['Whole_grain_products'],
            Age=data['Age'],
            Height=data['Height'],
            Body_Weight=data['Body_Weight'],
            Body_Weight_Re_Examination=data['Body_Weight_Re_Examination'],
            Heartburn=1
        )

        for symptom_name, res in all_results.items():
            prob = res['prob']
            percent = round(prob * 100, 1)

            safe_name = symptom_name.replace('_', '\\_')

            if percent > 50:
                res_header = f"✅ {safe_name} will disappear (probability: {percent}%)"
            else:
                res_header = f"⚠️ {safe_name} might remain (chance of disappearance: {percent}%)"

            rf_coefs = "\n".join([f"    - {k.replace('_', '\\_')}: {v}" for k, v in res['imp_rf'].items()])
            lr_coefs = "\n".join([f"    - {k.replace('_', '\\_')}: {v}" for k, v in res['coef_lr'].items()])

            symptom_msg = f"{res_header}\n\n"
            symptom_msg += f"Model: *RandomForestClassifier*\n"
            symptom_msg += f"📈 Impact coefficients:\n{rf_coefs}\n\n"
            symptom_msg += f"Model: *Logistic Regression*\n"
            symptom_msg += f"📈 Impact coefficients:\n{lr_coefs}\n"
            symptom_msg += f"\n--------------------------------------"

            bot.send_message(chat_id, symptom_msg, parse_mode="Markdown")

        edu_msg = (
            f"Model: *RandomForestClassifier*\n"
            f"*🔸Essence:* This is a 'consortium' of hundreds of independent decision trees. Each tree asks a series of questions (e.g., 'Is there swelling?', if yes — 'What is the pain level?'). The algorithm doesn't just add factors up, it analyzes their complex combinations. It works better with nonlinear data, where one factor might be important only in the presence of another.\n"
            f"*🔸Interpretation:* Based on Feature Importance. Example: The model might say that 'Physical activity level' is the most important factor, but it won't provide a simple linear formula. It shows a ranking: what influenced the prediction accuracy the most. This provides an understanding of the main priorities in the patient's rehabilitation.\n\n"
            f"Model: *Logistic Regression*\n\n"
            f"*🔸Essence:* This is a model that acts like a 'weighted scoring scale'. It assumes that each factor (e.g., age, injury type, protein level in diet) directly adds or subtracts chances of success. This is a classic statistical approach where we look for a direct link: 'the more of factor A, the higher the probability of result B'.\n"
            f"*🔸Interpretation:* Maximally transparent. The doctor sees the coefficients. Example: 'Every additional gram of whole grain in the diet reduces the risk of the symptom (insomnia) by a certain percentage'. This allows the doctor to clearly state exactly which parameter influences the patient and how strongly."
        )
        bot.send_message(chat_id, edu_msg, parse_mode="Markdown")

        markup = types.InlineKeyboardMarkup()
        btn_muscle = types.InlineKeyboardButton("💪 Calculate Muscle Mass", callback_data="predict_muscle")
        btn_restart = types.InlineKeyboardButton("🔄 Start Over", callback_data="restart")
        markup.row(btn_muscle, btn_restart)

        bot.send_message(chat_id, "Choose the next action:", reply_markup=markup)

    except Exception as e:
        bot.send_message(chat_id, f"An error occurred during analysis: {str(e)}")


@bot.callback_query_handler(func=lambda call: call.data == "predict_muscle")
def handle_muscle_prediction_click(call):
    chat_id = call.message.chat.id
    bot.answer_callback_query(call.id)

    data = patient_symptoms.get(chat_id)
    if not data or 'Shoulder_R_Final' not in data:
        bot.send_message(chat_id, "Data is outdated. Press /start.")
        return

    delta_weight = data['Body_Weight_Re_Examination'] - data['Body_Weight']
    delta_waist = data['Waist_Final'] - data['Waist_Start']
    delta_skinfat = data['SkinFat_Final'] - data['SkinFat_Start']

    primary_mass = Predict_Muscle_Mass_Primary(
        Whole_grain_products=data['Whole_grain_products'],
        Age=data['Age']
    )

    secondary_mass = Predict_Muscle_Mass_Secondary(
        Whole_grain_products=data['Whole_grain_products'],
        Age=data['Age'],
        Delta_Weight=delta_weight,
        Delta_Waist=delta_waist,
        Delta_Skinfold=delta_skinfat
    )

    if primary_mass is not None and secondary_mass is not None:
        result_text = (f"✅ *Predicted Muscle Mass (primary):* `{round(primary_mass, 2)}`\n\n"
                       f"✅ *Predicted Muscle Mass (secondary):* `{round(secondary_mass, 2)}`\n\n"
                       f"Do you want to account for amputation?")

        markup = types.InlineKeyboardMarkup()
        markup.add(types.InlineKeyboardButton("🦿 Yes, select amputation", callback_data="start_amputation_menu"))
        markup.add(types.InlineKeyboardButton("🔄 Start Over", callback_data="restart"))

        bot.send_message(chat_id, result_text, parse_mode="Markdown", reply_markup=markup)
    else:
        bot.send_message(chat_id, "Error: Not enough data for calculation.")


@bot.callback_query_handler(func=lambda call: call.data == "start_amputation_menu")
def handle_start_amputation_menu(call):
    chat_id = call.message.chat.id
    bot.answer_callback_query(call.id)

    patient_symptoms[chat_id]['selected_amps'] = []

    markup = generate_amputation_keyboard([])
    bot.send_message(chat_id, "Select amputated segments:", reply_markup=markup)


def generate_amputation_keyboard(selected_amps):
    markup = types.InlineKeyboardMarkup()
    segments = [
        ("Shoulder", "shoulder"), ("Forearm", "forearm"), ("Hand", "hand"),
        ("Thigh", "thigh"), ("Calf", "calf"), ("Foot", "foot")
    ]
    for name, code in segments:
        l_code = f"amp_{code}_l"
        r_code = f"amp_{code}_r"

        btn_text_l = f"✅ {name} (L)" if l_code in selected_amps else f"{name} (L)"
        btn_text_r = f"✅ {name} (R)" if r_code in selected_amps else f"{name} (R)"

        markup.row(
            types.InlineKeyboardButton(btn_text_l, callback_data=f"toggle_{l_code}"),
            types.InlineKeyboardButton(btn_text_r, callback_data=f"toggle_{r_code}")
        )

    if selected_amps:
        markup.add(types.InlineKeyboardButton("✅ Confirm Selection", callback_data="confirm_amputations"))
    markup.add(types.InlineKeyboardButton("🔄 Start Over", callback_data="restart"))
    return markup


@bot.callback_query_handler(func=lambda call: call.data.startswith("toggle_amp_"))
def handle_amputation_toggle(call):
    chat_id = call.message.chat.id
    amp_type = call.data.replace("toggle_", "")

    selected = patient_symptoms[chat_id].get('selected_amps', [])
    if amp_type in selected:
        selected.remove(amp_type)
    else:
        selected.append(amp_type)

    patient_symptoms[chat_id]['selected_amps'] = selected

    markup = generate_amputation_keyboard(selected)
    bot.edit_message_reply_markup(chat_id, call.message.message_id, reply_markup=markup)
    bot.answer_callback_query(call.id)


@bot.callback_query_handler(func=lambda call: call.data == "confirm_amputations")
def handle_confirm_amputations(call):
    chat_id = call.message.chat.id
    bot.answer_callback_query(call.id)

    data = patient_symptoms.get(chat_id, {})
    selected_amps = data.get('selected_amps', [])
    gender = data.get('Gender', 'male')

    if not selected_amps:
        bot.send_message(chat_id, "You did not select any amputation.")
        return

    total_lost_pct = 0.0
    for amp in selected_amps:
        parts = amp.split('_')
        if len(parts) >= 2:
            segment = parts[1]
            total_lost_pct += AMPUTATION_PERCENTAGES[gender].get(segment, 0.0)

    current_weight = data.get('Body_Weight_Re_Examination', data.get('Body_Weight', 80))
    height = data.get('Height', 1.75)

    if total_lost_pct < 100:
        corrected_weight = (current_weight / (100 - total_lost_pct)) * 100
    else:
        corrected_weight = current_weight

    corrected_bmi = corrected_weight / (height ** 2)
    lost_mass_kg = corrected_weight - current_weight
    patient_symptoms[chat_id]['Estimated_limb_mass_lost_kg'] = lost_mass_kg

    msg_text = (
        f"📊 *Anthropometry after amputation:*\n\n"
        f"Lost mass: `{round(total_lost_pct, 2)}%`\n"
        f"Estimated weight: `{round(corrected_weight, 2)} kg`\n"
        f"BMI: `{round(corrected_bmi, 2)}`\n"
        f"Mass of lost segments: `{round(lost_mass_kg, 2)} kg`\n\n"
        f"Enter the number of days after amputation:"
    )

    msg = bot.send_message(chat_id, msg_text, parse_mode="Markdown")
    bot.register_next_step_handler(msg, process_amp_days_and_predict)


def process_amp_days_and_predict(message):
    chat_id = message.chat.id
    try:
        patient_symptoms[chat_id]['Days_after_amputation'] = float(message.text)

        data = patient_symptoms.get(chat_id)
        selected_amps = data.get('selected_amps', [])

        is_upper = any(seg in amp for amp in selected_amps for seg in ['shoulder', 'forearm', 'hand'])
        level = "upper_limb" if is_upper else "lower_limb"

        df = pd.read_csv("Rehabilitation_imputed_whole_grain.csv")
        X, y, feat_cols = prepare_amputation_dataset_v2(
            df, side_available="R", target_mode="fallback", use_weight_corrected=True
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model = RandomForestRegressor(n_estimators=500, random_state=42)
        model.fit(X_train, y_train)

        delta_waist = data['Waist_Final'] - data['Waist_Start']
        delta_skinfat = data['SkinFat_Final'] - data['SkinFat_Start']

        new_person = {
            "Whole_grain_products": data['Whole_grain_products'],
            "Age": data['Age'],
            "Delta_Skinfold": delta_skinfat,
            "Delta_Waist": delta_waist,
            "Days_after_amputation": data['Days_after_amputation'],
            "Amputation_level": level,
            "Estimated_limb_mass_lost_kg": data['Estimated_limb_mass_lost_kg'],
            "Body_Weight": data['Body_Weight'],
            "Body_Weight_Re_Examination": data['Body_Weight_Re_Examination']
        }

        new_X = build_new_person_features_v2(new_person, feat_cols)
        pred = model.predict(new_X)[0]

        result_text = f"✅ *Final Muscle Mass Prediction after amputation:* `{round(pred, 2)}`"

        markup = types.InlineKeyboardMarkup()
        markup.add(types.InlineKeyboardButton("🔄 Start Over", callback_data="restart"))

        bot.send_message(chat_id, result_text, parse_mode="Markdown", reply_markup=markup)

    except Exception as e:
         msg = bot.send_message(chat_id, f"Enter a number (number of days):")
         bot.register_next_step_handler(msg, process_amp_days_and_predict)


@bot.callback_query_handler(func=lambda call: call.data == "restart")
def handle_restart_click(call):
    bot.answer_callback_query(call.id)
    start_message(call.message)

if __name__ == '__main__':
    bot.polling(none_stop=True)
