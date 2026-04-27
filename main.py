import calendar
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import telebot
from telebot import types
from telebot.apihelper import ApiTelegramException
import math

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

FRISANCHO_UAMC = {
    'male': {
        (1, 2): [110, 113, 119, 127, 135, 144, 147],
        (2, 3): [111, 114, 122, 130, 140, 146, 150],
        (3, 4): [117, 123, 131, 137, 143, 148, 153],
        (4, 5): [123, 126, 133, 141, 148, 156, 159],
        (5, 6): [128, 133, 140, 147, 154, 162, 169],
        (6, 7): [131, 135, 142, 151, 161, 170, 177],
        (7, 8): [137, 139, 151, 160, 168, 177, 190],
        (8, 9): [140, 145, 154, 162, 170, 182, 187],
        (9, 10): [151, 154, 161, 170, 183, 196, 202],
        (10, 11): [156, 160, 166, 180, 191, 209, 221],
        (11, 12): [159, 165, 173, 183, 195, 205, 230],
        (12, 13): [167, 171, 182, 195, 210, 223, 241],
        (13, 14): [172, 179, 196, 211, 226, 238, 245],
        (14, 15): [189, 199, 212, 223, 240, 260, 264],
        (15, 16): [199, 204, 218, 237, 254, 266, 272],
        (16, 17): [213, 225, 234, 249, 269, 287, 296],
        (17, 18): [224, 231, 245, 258, 273, 294, 312],
        (18, 19): [226, 237, 252, 264, 283, 298, 324],
        (19, 25): [238, 245, 257, 273, 289, 309, 321],
        (25, 35): [243, 250, 264, 279, 298, 314, 326],
        (35, 45): [247, 255, 269, 286, 302, 318, 327],
        (45, 55): [239, 249, 265, 281, 300, 315, 326],
        (55, 65): [236, 245, 260, 278, 295, 310, 320],
        (65, 200):  [223, 235, 251, 268, 284, 298, 306]
    },
    'female': {
        (1, 2): [105, 111, 117, 124, 132, 139, 143],
        (2, 3): [111, 114, 119, 126, 133, 142, 147],
        (3, 4): [113, 119, 124, 132, 140, 146, 152],
        (4, 5): [115, 121, 128, 136, 144, 152, 157],
        (5, 6): [125, 128, 134, 142, 151, 159, 165],
        (6, 7): [130, 133, 138, 145, 154, 166, 171],
        (7, 8): [129, 135, 142, 151, 160, 171, 176],
        (8, 9): [138, 140, 151, 160, 171, 183, 194],
        (9, 10): [147, 150, 158, 167, 180, 194, 198],
        (10, 11): [148, 150, 159, 170, 180, 190, 197],
        (11, 12): [150, 158, 171, 181, 196, 217, 223],
        (12, 13): [162, 166, 180, 191, 201, 214, 220],
        (13, 14): [169, 175, 183, 198, 211, 226, 240],
        (14, 15): [174, 179, 190, 201, 216, 232, 247],
        (15, 16): [175, 178, 189, 202, 215, 228, 244],
        (16, 17): [170, 180, 190, 202, 216, 234, 249],
        (17, 18): [175, 183, 194, 205, 221, 239, 257],
        (18, 19): [174, 179, 191, 202, 215, 237, 245],
        (19, 25): [179, 185, 195, 207, 221, 236, 249],
        (25, 35): [183, 188, 199, 212, 228, 246, 264],
        (35, 45): [186, 192, 205, 218, 236, 257, 272],
        (45, 55): [187, 193, 206, 220, 238, 260, 274],
        (55, 65): [187, 196, 209, 225, 244, 266, 280],
        (65, 200):  [185, 195, 208, 225, 244, 264, 279]
    }
}

def get_bmi_interpretation(weight, height_m):
    bmi = weight / (height_m ** 2)
    if bmi < 18.5: text = "Underweight (Depletion Risk)"
    elif 18.5 <= bmi <= 24.9: text = "Normal"
    elif 25 <= bmi <= 29.9: text = "Overweight"
    else: text = "Obese"
    return round(bmi, 1), text

def safe_parse_date(date_str):
    if pd.isna(date_str):
        return pd.NaT
    s = str(date_str).strip()
    if "." not in s:
        return pd.NaT
    parts = s.split(".")
    if len(parts) != 3:
        return pd.NaT
    try:
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        last_day = calendar.monthrange(year, month)[1]
        day = min(day, last_day)
        return pd.Timestamp(year=year, month=month, day=day)
    except:
        return pd.NaT

def clean_float_string(s):
    s = str(s).strip()
    if s.endswith(".0") and s.count(".") == 1:
        return s[:-2]
    return s

def evaluate_anthropometry(gender, age, mac_cm, tsf_mm):
    try:
        age = float(age)
        mac_cm = float(mac_cm)
        tsf_mm = float(tsf_mm)

        mac_mm = mac_cm * 10.0
        uamc_mm = mac_mm - (math.pi * tsf_mm)

        gender_key = 'male' if gender.lower() in ['male', 'm'] else 'female'
        table = FRISANCHO_UAMC[gender_key]

        selected_row = table[(65, 200)]
        for (min_age, max_age), row in table.items():
            if min_age <= age < max_age:
                selected_row = row
                break

        p5, p10, p25, p50, p75, p90, p95 = selected_row

        p15 = p10 + (p25 - p10) * 0.3333
        p85 = p75 + (p90 - p75) * 0.6667

        if uamc_mm > p95:
            status = "High"
        elif p85 < uamc_mm <= p95:
            status = "Elevated"
        elif p15 <= uamc_mm <= p85:
            status = "Optimal"
        elif p5 < uamc_mm < p15:
            status = "Significant deficit"
        else:
            status = "Severe depletion (Sarcopenia)"

        return {
            "uamc_mm": round(uamc_mm, 1),
            "nutritional_status": status
        }
    except Exception as e:
        return None

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
        df["Functional_muscle_change_index"] = df[["Delta_Shoulder", "Delta_Grip"]].sum(axis=1, min_count=1)
    elif target_mode == "fallback":
        df["Functional_muscle_change_index"] = df[["Delta_Grip", "Delta_Shoulder"]].sum(axis=1, min_count=1)
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
    y = df["Functional_muscle_change_index"]

    if dropna:
        must_have = ["Whole_grain_products", "Age", "Delta_Skinfold", "Delta_Weight_corrected", "Delta_Waist", "Functional_muscle_change_index"]
        data = pd.concat([X, y], axis=1).dropna(subset=must_have)
        X = data[feat_cols]
        y = data["Functional_muscle_change_index"]

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

def Gastrointestinal_Tract_Symptoms(Whole_grain_products, Age, Height, Body_Weight, Body_Weight_Re_Examination):
    try:
        df = pd.read_csv("Rehabilitation_imputed_whole_grain_timeframe.csv")
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
        'Constipation_Re_Examination': 'Constipation',
        'Bloating_Re_Examination': 'Bloating',
        'Insomnia_Re_Examination': 'Insomnia'
    }

    cols_to_clean = ["Whole_grain_products", "Age", "Height", "Body_Weight", "Body_Weight_Re_Examination"]
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

            cleaned_col = temp_df[col_name].apply(clean_float_string)
            temp_df["Symptom_resolution_status"] = (cleaned_col != "0").astype(int)

            if label in temp_df.columns:
                had_symptom = temp_df[label].apply(clean_float_string) != "0"
                temp_df = temp_df[had_symptom]

            X = temp_df[features]
            y = temp_df["Symptom_resolution_status"]

            if len(y.unique()) < 2:
                default_prob = float(y.iloc[0]) if not y.empty else 0.0
                results_dict[label] = {
                    "prob": default_prob,
                    "imp_rf": {f: 0 for f in features},
                    "coef_lr": {f: 0 for f in features}
                }
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            model_rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
            model_rf.fit(X_train, y_train)

            model_lr = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
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
        df = pd.read_csv("Rehabilitation_imputed_whole_grain_timeframe.csv")
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
    df["Functional_muscle_change_index"] = df["Delta_Shoulder"] + df["Delta_Grip"]

    X = df[["Whole_grain_products", "Age"]]
    y = df["Functional_muscle_change_index"]

    data = pd.concat([X, y], axis=1).dropna()
    if data.empty:
        return None

    X = data[["Whole_grain_products", "Age"]]
    y = data["Functional_muscle_change_index"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=400, random_state=42)
    model.fit(X_train, y_train)

    new_person = pd.DataFrame([{"Whole_grain_products": Whole_grain_products, "Age": Age}])
    return model.predict(new_person)[0]

def Predict_Muscle_Mass_Secondary(Whole_grain_products, Age, Delta_Weight, Delta_Waist, Delta_Skinfold):
    try:
        df = pd.read_csv("Rehabilitation_imputed_whole_grain_timeframe.csv")
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
    df["Functional_muscle_change_index"] = df["Delta_Shoulder"] + df["Delta_Grip"]

    df["Delta_Skinfold"] = df["ThicknessSkinFatParaumbilicalAreaRectusAbdominisMuscle_Re_Examination"] - df["ThicknessSkinFatParaumbilicalAreaRectusAbdominisMuscle"]
    df["Delta_Weight"] = df["Body_Weight_Re_Examination"] - df["Body_Weight"]
    df["Delta_Waist"] = df["WaistCircumference_Re_Examination"] - df["WaistCircumference"]

    features = ["Whole_grain_products", "Age", "Delta_Skinfold", "Delta_Weight", "Delta_Waist"]
    data = df[features + ["Functional_muscle_change_index"]].dropna()

    if data.empty:
        return None

    X = data[features]
    y = data["Functional_muscle_change_index"]

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

def Predict_Symptom_Time(symptom_col_reexam, symptom_col_start, Whole_grain_products, Age, Body_Weight, date_of_examination, date_of_re_examination):
    try:
        df = pd.read_csv("Rehabilitation_imputed_whole_grain_timeframe.csv")
    except FileNotFoundError:
        return None

    if "Date_Of_Examination" not in df.columns or "Date_of_Re_Examination" not in df.columns or symptom_col_reexam not in df.columns:
        return None

    df["exam_date"] = df["Date_Of_Examination"].apply(safe_parse_date)
    df["reexam_date"] = df["Date_of_Re_Examination"].apply(safe_parse_date)

    def compute_days(row):
        start_val = clean_float_string(row.get(symptom_col_start, "0"))
        reexam_val = clean_float_string(row.get(symptom_col_reexam, "0"))

        if start_val == "0" or reexam_val == "0":
            return np.nan

        s_date = safe_parse_date(reexam_val)
        e_date = row["exam_date"]
        if pd.notna(s_date) and pd.notna(e_date):
            delta = (s_date - e_date).days
            if delta >= 0:
                return delta
        return np.nan

    df["days_to_disappearance"] = df.apply(compute_days, axis=1)
    df["planned_rehab_days"] = (df["reexam_date"] - df["exam_date"]).dt.days

    train_df = df.dropna(subset=["days_to_disappearance"]).copy()

    for col in ["Whole_grain_products", "Age", "Body_Weight"]:
        if col in train_df.columns:
            if train_df[col].dtype == object:
                train_df[col] = train_df[col].astype(str).str.replace(",", ".", regex=False)
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

    features = ["planned_rehab_days", "Age", "Whole_grain_products", "Body_Weight"]
    train_df = train_df.dropna(subset=features)

    if train_df.empty:
        return None

    X = train_df[features]
    y = train_df["days_to_disappearance"]

    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=5)
    model.fit(X, y)

    ex_d = safe_parse_date(date_of_examination)
    rex_d = safe_parse_date(date_of_re_examination)
    if pd.isna(ex_d) or pd.isna(rex_d):
        return None

    planned_days = (rex_d - ex_d).days
    if planned_days <= 0:
        return None

    new_person = pd.DataFrame([{
        "planned_rehab_days": planned_days,
        "Age": Age,
        "Whole_grain_products": Whole_grain_products,
        "Body_Weight": Body_Weight
    }])

    pred_days = model.predict(new_person)[0]
    pred_days = max(0, min(round(pred_days), planned_days))
    pred_date = ex_d + pd.Timedelta(days=pred_days)

    return {
        "days": pred_days,
        "date": pred_date.strftime("%d.%m.%Y"),
        "planned": planned_days
    }

def get_evidence_base_text():
    return (
        "*Clinical Interpretation & Evidence Base:*\n"
        "- *Nutritional Assessment:* Based on UAMC standard percentiles (Frisancho, 1990).\n"
        "- *Muscle Change Index:* ML estimation strictly relies on baseline dynamic metrics.\n"
        "- *Dietary Guidelines:* Calibrations align with ESPEN standards for clinical nutrition.\n\n"
        "⚠️ *Notice: This tool provides statistical ML estimations to support recovery tracking and does NOT replace professional clinical judgment.*"
    )

bot = telebot.TeleBot('8464210577:AAHrEPRdNsgluESEIb1A9VdrYQnm_SFQXFo')
patient_symptoms = {}

@bot.message_handler(commands=['start'])
def start_message(message):
    chat_id = message.chat.id

    if chat_id in patient_symptoms:
        if 'last_menu_id' in patient_symptoms[chat_id]:
            try:
                bot.delete_message(chat_id, patient_symptoms[chat_id]['last_menu_id'])
            except ApiTelegramException:
                pass
                
        bot.send_message(chat_id, "Попередній процес скасовано починаємо спочатку")
        
    bot.clear_step_handler_by_chat_id(chat_id)
    
    patient_symptoms[chat_id] = {'snaq_score': 0}
    
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    markup.add("Male", "Female")

    msg = bot.send_message(
        chat_id,
        "Hello! 👋\nI am SmartRecover AI, a bot for muscle strength and motor coordination recovery. I analyze progress and adapt loads to your condition. I provide personalized treatment plans and nutritional advice.\n\nPlease select your gender:",
        reply_markup=markup
    )
    patient_symptoms[chat_id]['last_menu_id'] = msg.message_id
    
    bot.register_next_step_handler(msg, get_gender)

def get_gender(message):
    if message.text == "Male":
        patient_symptoms[message.chat.id]['Gender'] = 'male'
    elif message.text == "Female":
        patient_symptoms[message.chat.id]['Gender'] = 'female'
    else:
        patient_symptoms[message.chat.id]['Gender'] = 'male'

    markup = types.ReplyKeyboardRemove()
    msg = bot.send_message(message.chat.id, "Enter age (full years):", reply_markup=markup)
    bot.register_next_step_handler(msg, get_age)

def get_age(message):
    if message.text == '/start':
        start_message(message)
        return
    try:
        patient_symptoms[message.chat.id]['Age'] = int(message.text)
        
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        markup.add("At the beginning", "In progress", "Completed")
        msg = bot.send_message(message.chat.id, "Select current stage:", reply_markup=markup)
        bot.register_next_step_handler(msg, get_rehab_stage)
    except ValueError:
        msg = bot.send_message(message.chat.id, "Please enter a valid integer.")
        bot.register_next_step_handler(msg, get_age)

def get_rehab_stage(message):
    stage = message.text
    if stage not in ["At the beginning", "In progress", "Completed"]:
        stage = "At the beginning"
    
    patient_symptoms[message.chat.id]['Stage'] = stage
    markup = types.ReplyKeyboardRemove()
    
    msg = bot.send_message(message.chat.id, "Enter START date (DD.MM.YYYY):", reply_markup=markup)
    bot.register_next_step_handler(msg, get_start_date)

def get_start_date(message):
    chat_id = message.chat.id
    patient_symptoms[chat_id]['exam_date'] = message.text.strip()
    stage = patient_symptoms[chat_id]['Stage']
    
    if stage == "At the beginning":
        start_snaq_question_1(chat_id)
    elif stage == "In progress":
        msg = bot.send_message(chat_id, "Enter CURRENT date (DD.MM.YYYY):")
        bot.register_next_step_handler(msg, get_end_date)
    else:
        msg = bot.send_message(chat_id, "Enter PLANNED END date (DD.MM.YYYY):")
        bot.register_next_step_handler(msg, get_end_date)

def get_end_date(message):
    patient_symptoms[message.chat.id]['reexam_date'] = message.text.strip()
    start_snaq_question_1(message.chat.id)

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

    try:
        bot.edit_message_text("SNAQ Questionnaire (2/4): How does the food taste to you?", chat_id, call.message.message_id, reply_markup=markup)
    except ApiTelegramException as e:
        if "message is not modified" not in str(e):
            raise e

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
    markup.add(types.InlineKeyboardButton("I rarely eat enough to feel full", callback_data="snaq3_5"))

    try:
        bot.edit_message_text("SNAQ Questionnaire (3/4): When I eat, I feel...", chat_id, call.message.message_id, reply_markup=markup)
    except ApiTelegramException as e:
        if "message is not modified" not in str(e):
            raise e

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

    try:
        bot.edit_message_text("SNAQ Questionnaire (4/4): Usually, in a day I have...", chat_id, call.message.message_id, reply_markup=markup)
    except ApiTelegramException as e:
        if "message is not modified" not in str(e):
            raise e

@bot.callback_query_handler(func=lambda call: call.data.startswith("snaq4_"))
def handle_snaq_4(call):
    chat_id = call.message.chat.id
    score = int(call.data.split('_')[1])
    patient_symptoms[chat_id]['snaq_score'] += score

    total_score = patient_symptoms[chat_id]['snaq_score']
    if total_score <= 14:
        bot.send_message(chat_id, f"⚠️ *ATTENTION! SNAQ score: {total_score}*.\nSignificant risk of weight loss and malnutrition. Dietitian consultation recommended.", parse_mode="Markdown")
    else:
        bot.send_message(chat_id, f"✅ *SNAQ score: {total_score}*.\nLow malnutrition risk.", parse_mode="Markdown")

    msg = bot.send_message(chat_id, "Enter whole grain products amount (g/day):")
    bot.register_next_step_handler(msg, get_grains)

def get_grains(message):
    try:
        val = float(message.text.replace(',', '.'))
        patient_symptoms[message.chat.id]['Whole_grain_products'] = val

        msg = bot.send_message(
            message.chat.id,
            "Enter height (m or cm, e.g., 1.75 or 175).\n\n"
            "❗️ *Amputees:* enter `0` to estimate via knee height.",
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
            "To estimate height, measure *Knee Height (KH)*:\n"
            "Bend knee at 90°, measure from heel to anterior thigh surface (above kneecap).\n\n"
            "Enter value in *centimeters* (e.g., 52):",
            parse_mode="Markdown"
        )
        bot.register_next_step_handler(msg, calculate_alternative_height)
        return

    try:
        val = float(text)
        if val > 3.0: 
            val = val / 100.0 # Auto-convert cm to m
        patient_symptoms[chat_id]['Height'] = val
        msg = bot.send_message(chat_id, "Thank you! Enter INITIAL weight (kg):")
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

        bot.send_message(chat_id, f"✅ Estimated height: *{round(height_m, 2)} m*", parse_mode="Markdown")

        msg = bot.send_message(chat_id, "Enter INITIAL weight (kg):")
        bot.register_next_step_handler(msg, get_weight_start)
    except ValueError:
        msg = bot.send_message(chat_id, "Please enter a valid number (cm).")
        bot.register_next_step_handler(msg, calculate_alternative_height)

def get_weight_start(message):
    chat_id = message.chat.id
    try:
        val = float(message.text.replace(',', '.'))
        if val <= 0: raise ValueError
        patient_symptoms[chat_id]['Body_Weight'] = val
        
        if patient_symptoms[chat_id]['Stage'] == "At the beginning":
            msg = bot.send_message(chat_id, "Enter INITIAL waist circumference (cm):")
            bot.register_next_step_handler(msg, get_waist_start)
        else:
            msg = bot.send_message(chat_id, "Enter CURRENT/FINAL weight (kg):")
            bot.register_next_step_handler(msg, get_weight_final)
    except ValueError:
        bot.send_message(chat_id, "Please enter a positive number.")
        bot.register_next_step_handler(message, get_weight_start)

def get_weight_final(message):
    chat_id = message.chat.id
    try:
        val = float(message.text.replace(',', '.'))
        if val <= 0: raise ValueError
        patient_symptoms[chat_id]['Body_Weight_Re_Examination'] = val
        msg = bot.send_message(chat_id, "Enter INITIAL waist circumference (cm):")
        bot.register_next_step_handler(msg, get_waist_start)
    except ValueError:
        bot.send_message(chat_id, "Please enter a valid number.")
        bot.register_next_step_handler(message, get_weight_final)

def get_waist_start(message):
    chat_id = message.chat.id
    try:
        val = float(message.text.replace(',', '.'))
        patient_symptoms[chat_id]['Waist_Start'] = val
        
        if patient_symptoms[chat_id]['Stage'] == "At the beginning":
            msg = bot.send_message(chat_id, "Enter INITIAL Triceps skinfold thickness (mm):")
            bot.register_next_step_handler(msg, get_triceps_start)
        else:
            msg = bot.send_message(chat_id, "Enter CURRENT/FINAL waist circumference (cm):")
            bot.register_next_step_handler(msg, get_waist_final)
    except ValueError:
        bot.send_message(chat_id, "Please enter a valid number.")
        bot.register_next_step_handler(message, get_waist_start)

def get_waist_final(message):
    try:
        val = float(message.text.replace(',', '.'))
        patient_symptoms[message.chat.id]['Waist_Final'] = val
        msg = bot.send_message(message.chat.id, "Enter INITIAL Triceps skinfold thickness (mm):")
        bot.register_next_step_handler(msg, get_triceps_start)
    except ValueError:
        bot.send_message(message.chat.id, "Please enter a valid number.")
        bot.register_next_step_handler(message, get_waist_final)

def get_triceps_start(message):
    chat_id = message.chat.id
    try:
        val = float(message.text.replace(',', '.'))
        patient_symptoms[chat_id]['Triceps_Start'] = val
        
        if patient_symptoms[chat_id]['Stage'] == "At the beginning":
            msg = bot.send_message(chat_id, "Enter INITIAL Paraumbilical skinfold thickness (mm):")
            bot.register_next_step_handler(msg, get_skinfat_start)
        else:
            msg = bot.send_message(chat_id, "Enter CURRENT/FINAL Triceps skinfold thickness (mm):")
            bot.register_next_step_handler(msg, get_triceps_final)
    except ValueError:
        bot.send_message(chat_id, "Please enter a valid number.")
        bot.register_next_step_handler(message, get_triceps_start)

def get_triceps_final(message):
    try:
        val = float(message.text.replace(',', '.'))
        patient_symptoms[message.chat.id]['Triceps_Final'] = val
        msg = bot.send_message(message.chat.id, "Enter INITIAL Paraumbilical skinfold thickness (mm):")
        bot.register_next_step_handler(msg, get_skinfat_start)
    except ValueError:
        bot.send_message(message.chat.id, "Please enter a valid number.")
        bot.register_next_step_handler(message, get_triceps_final)

def get_skinfat_start(message):
    chat_id = message.chat.id
    try:
        val = float(message.text.replace(',', '.'))
        patient_symptoms[chat_id]['SkinFat_Start'] = val
        
        if patient_symptoms[chat_id]['Stage'] == "At the beginning":
            msg = bot.send_message(chat_id, "Enter INITIAL circumference of the right upper arm (cm):")
            bot.register_next_step_handler(msg, get_shoulder_start)
        else:
            msg = bot.send_message(chat_id, "Enter CURRENT/FINAL Paraumbilical skinfold thickness (mm):")
            bot.register_next_step_handler(msg, get_skinfat_final)
    except ValueError:
        bot.send_message(chat_id, "Please enter a valid number.")
        bot.register_next_step_handler(message, get_skinfat_start)

def get_skinfat_final(message):
    try:
        val = float(message.text.replace(',', '.'))
        patient_symptoms[message.chat.id]['SkinFat_Final'] = val
        msg = bot.send_message(message.chat.id, "Enter INITIAL circumference of the right upper arm (cm):")
        bot.register_next_step_handler(msg, get_shoulder_start)
    except ValueError:
        bot.send_message(message.chat.id, "Please enter a valid number.")
        bot.register_next_step_handler(message, get_skinfat_final)

def get_shoulder_start(message):
    chat_id = message.chat.id
    try:
        val = float(message.text.replace(',', '.'))
        patient_symptoms[chat_id]['Shoulder_R_Start'] = val
        
        if patient_symptoms[chat_id]['Stage'] == "At the beginning":
            perform_initial_assessment(chat_id)
        else:
            msg = bot.send_message(chat_id, "Enter CURRENT/FINAL circumference of the right upper arm (cm):")
            bot.register_next_step_handler(msg, get_shoulder_final)
    except ValueError:
        bot.send_message(chat_id, "Please enter a valid number.")
        bot.register_next_step_handler(message, get_shoulder_start)

def get_shoulder_final(message):
    try:
        val = float(message.text.replace(',', '.'))
        patient_symptoms[message.chat.id]['Shoulder_R_Final'] = val
        perform_prediction(message.chat.id)
    except ValueError:
        bot.send_message(message.chat.id, "Please enter a valid number.")
        bot.register_next_step_handler(message, get_shoulder_final)

def send_uamc_explanation(chat_id):
    uamc_text = (
        "*Upper Arm Muscle Circumference (UAMC) Calculation*\n"
        "Formula: `UAMC = MAC - (π × TSF)`\n\n"
        "Estimates lean muscle mass by subtracting fat.\n"
        "- *UAMC*: Upper Arm Muscle Circumference (mm).\n"
        "- *MAC*: Mid-Arm Circumference (mm).\n"
        "- *TSF*: Triceps Skinfold thickness (mm).\n"
        "- *π*: Pi (≈ 3.14159)."
    )
    bot.send_message(chat_id, uamc_text, parse_mode="Markdown")

def perform_initial_assessment(chat_id):
    try:
        data = patient_symptoms[chat_id]
        
        send_uamc_explanation(chat_id)

        anthro_eval_start = evaluate_anthropometry(
            gender=data.get('Gender', 'male'),
            age=data.get('Age', 30),
            mac_cm=data.get('Shoulder_R_Start', 0),
            tsf_mm=data.get('Triceps_Start', 0)
        )

        diagnosis_en = {
            "High": "High muscle mass level 💪 (excellent reserve)",
            "Elevated": "Well-developed muscles 👍 (above average)",
            "Optimal": "Optimal level ✅ (healthy norm)",
            "Significant deficit": "Significant muscle deficit ⚠️ (increased protein nutrition required)",
            "Severe depletion (Sarcopenia)": "Severe depletion / Sarcopenia ❌ (critical muscle loss)"
        }

        if anthro_eval_start:
            bmi_val, bmi_status = get_bmi_interpretation(data['Body_Weight'], data['Height'])
            
            anthro_msg = (
                f"📊 *BASELINE ASSESSMENT:*\n\n"
                f"BMI: `{bmi_val}` - {bmi_status}\n\n"
                f"Upper Arm Muscle Circumference (UAMC): `{anthro_eval_start['uamc_mm']} mm`\n"
                f"Nutritional Status: *{diagnosis_en.get(anthro_eval_start['nutritional_status'], anthro_eval_start['nutritional_status'])}*\n\n"
            )
            anthro_msg += get_evidence_base_text()
            bot.send_message(chat_id, anthro_msg, parse_mode="Markdown")
        
        markup = types.InlineKeyboardMarkup()
        markup.add(types.InlineKeyboardButton("🔄 Start Over", callback_data="restart"))
        bot.send_message(chat_id, "Dynamics prediction is unavailable at the initial stage.", reply_markup=markup)

    except Exception as e:
        bot.send_message(chat_id, f"An error occurred during initial analysis: {str(e)}")

def perform_prediction(chat_id):
    try:
        data = patient_symptoms[chat_id]

        all_results = Gastrointestinal_Tract_Symptoms(
            Whole_grain_products=data['Whole_grain_products'],
            Age=data['Age'],
            Height=data['Height'],
            Body_Weight=data['Body_Weight'],
            Body_Weight_Re_Examination=data['Body_Weight_Re_Examination']
        )

        symptom_columns = {
            'Constipation': ('Constipation_Re_Examination', 'Constipation'),
            'Bloating': ('Bloating_Re_Examination', 'Bloating'),
            'Insomnia': ('Insomnia_Re_Examination', 'Insomnia')
        }

        for symptom_name, res in all_results.items():
            prob = res.get('prob', 0)
            percent = round(prob * 100, 1)
            safe_name = symptom_name.replace('_', '\\_')

            if percent > 50:
                res_header = f"✅ {safe_name} will disappear (probability: {percent}%)"
            else:
                res_header = f"⚠️ {safe_name} might remain (chance of disappearance: {percent}%)"

            symptom_msg = f"{res_header}\n\n"

            if percent > 50 and symptom_name in symptom_columns:
                reexam_col, start_col = symptom_columns[symptom_name]
                symp_time = Predict_Symptom_Time(
                    symptom_col_reexam=reexam_col,
                    symptom_col_start=start_col,
                    Whole_grain_products=data.get('Whole_grain_products', 0),
                    Age=data.get('Age', 0),
                    Body_Weight=data.get('Body_Weight', 0),
                    date_of_examination=data.get('exam_date', ''),
                    date_of_re_examination=data.get('reexam_date', '')
                )

                if symp_time:
                    if symp_time['days'] <= symp_time['planned']:
                        symptom_msg += (
                            f"⏳ *{safe_name} Disappearance Timeframe:*\n"
                            f"Planned rehab duration: `{symp_time['planned']} days`\n"
                            f"Expected to disappear in: *{symp_time['days']} days*\n"
                            f"Estimated date: *{symp_time['date']}*\n\n"
                        )
                    else:
                        symptom_msg += (
                            f"⏳ *{safe_name} Disappearance Timeframe:*\n"
                            f"Planned rehab duration: `{symp_time['planned']} days`\n"
                            f"⚠️ *Attention:* Symptom may persist beyond planned discharge.\n"
                            f"Expected to disappear in: *{symp_time['days']} days*\n"
                            f"Estimated date: *{symp_time['date']}*\n\n"
                        )

            rf_coefs = "\n".join([f"    - {k.replace('_', '\\_')}: {v}" for k, v in res.get('imp_rf', {}).items()])
            lr_coefs = "\n".join([f"    - {k.replace('_', '\\_')}: {v}" for k, v in res.get('coef_lr', {}).items()])

            symptom_msg += f"Model: *RandomForestClassifier*\n"
            symptom_msg += f"📈 Impact coefficients:\n{rf_coefs}\n\n"
            symptom_msg += f"Model: *Logistic Regression*\n"
            symptom_msg += f"📈 Impact coefficients:\n{lr_coefs}\n"
            symptom_msg += f"\n--------------------------------------"

            bot.send_message(chat_id, symptom_msg, parse_mode="Markdown")

        edu_msg = (
            f"Model: *RandomForestClassifier*\n\n"
            f"🔸*Essence:* A 'consortium' of decision trees analyzing complex, nonlinear factor combinations.\n"
            f"🔸*Interpretation:* Shows what influenced prediction accuracy the most, helping prioritize rehab goals.\n\n"
            f"Model: *Logistic Regression*\n\n"
            f"🔸*Essence:* A 'weighted scoring scale' determining how each factor directly affects success.\n"
            f"🔸*Interpretation:* Highly transparent, revealing the exact percentage risk reduction per factor."
        )
        bot.send_message(chat_id, edu_msg, parse_mode="Markdown")

        send_uamc_explanation(chat_id)

        anthro_eval_start = evaluate_anthropometry(
            gender=data.get('Gender', 'male'),
            age=data.get('Age', 30),
            mac_cm=data.get('Shoulder_R_Start', 0),
            tsf_mm=data.get('Triceps_Start', 0)
        )

        anthro_eval_final = evaluate_anthropometry(
            gender=data.get('Gender', 'male'),
            age=data.get('Age', 30),
            mac_cm=data.get('Shoulder_R_Final', 0),
            tsf_mm=data.get('Triceps_Final', 0)
        )

        diagnosis_en = {
            "High": "High muscle mass level 💪 (excellent reserve)",
            "Elevated": "Well-developed muscles 👍 (above average)",
            "Optimal": "Optimal level ✅ (healthy norm)",
            "Significant deficit": "Significant muscle deficit ⚠️ (increased protein nutrition required)",
            "Severe depletion (Sarcopenia)": "Severe depletion / Sarcopenia ❌ (critical muscle loss)"
        }

        if anthro_eval_start and anthro_eval_final:
            bmi_val, bmi_status = get_bmi_interpretation(data['Body_Weight'], data['Height'])

            anthro_msg = (
                f"📊 *Lean Muscle Mass Assessment:*\n\n"
                f"BMI: `{bmi_val}` - {bmi_status}\n\n"
                f"🔹 *BEFORE Rehabilitation:*\n"
                f"Upper Arm Muscle Circumference: `{anthro_eval_start['uamc_mm']} mm`\n"
                f"Nutritional Status: *{diagnosis_en.get(anthro_eval_start['nutritional_status'], anthro_eval_start['nutritional_status'])}*\n\n"
                f"🔹 *AFTER/CURRENT Rehabilitation:*\n"
                f"Upper Arm Muscle Circumference: `{anthro_eval_final['uamc_mm']} mm`\n"
                f"Nutritional Status: *{diagnosis_en.get(anthro_eval_final['nutritional_status'], anthro_eval_final['nutritional_status'])}*\n\n"
            )
            anthro_msg += get_evidence_base_text()
            bot.send_message(chat_id, anthro_msg, parse_mode="Markdown")

        markup = types.InlineKeyboardMarkup()
        markup.add(types.InlineKeyboardButton("💪 Calculate Muscle Mass", callback_data="predict_muscle"))
        markup.add(types.InlineKeyboardButton("🔄 Start Over", callback_data="restart"))

        bot.send_message(chat_id, "Choose the next action:", reply_markup=markup)

    except Exception as e:
        bot.send_message(chat_id, f"An error occurred during analysis: {str(e)}")


@bot.callback_query_handler(func=lambda call: call.data == "predict_muscle")
def handle_muscle_prediction_click(call):
    try:
        bot.answer_callback_query(call.id)
    except ApiTelegramException:
        pass

    chat_id = call.message.chat.id
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
        result_text = (f"✅ *Predicted Functional Muscle Change Index (primary):* `{round(primary_mass, 2)}` index units\n"
                       f"*(Reflects functional muscle mass dynamics)*\n\n"
                       f"✅ *Predicted Functional Muscle Change Index (secondary):* `{round(secondary_mass, 2)}` index units\n\n")

        result_text += f"Do you want to account for amputation?"

        markup = types.InlineKeyboardMarkup()
        markup.add(types.InlineKeyboardButton("🦿 Yes, select amputation", callback_data="start_amputation_menu"))
        markup.add(types.InlineKeyboardButton("🔄 Start Over", callback_data="restart"))

        bot.send_message(chat_id, result_text, parse_mode="Markdown", reply_markup=markup)
    else:
        bot.send_message(chat_id, "Error: Not enough data for calculation.")


@bot.callback_query_handler(func=lambda call: call.data == "start_amputation_menu")
def handle_start_amputation_menu(call):
    try:
        bot.answer_callback_query(call.id)
    except ApiTelegramException:
        pass

    chat_id = call.message.chat.id
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
    try:
        bot.answer_callback_query(call.id)
    except ApiTelegramException:
        pass

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


@bot.callback_query_handler(func=lambda call: call.data == "confirm_amputations")
def handle_confirm_amputations(call):
    try:
        bot.answer_callback_query(call.id)
    except ApiTelegramException:
        pass

    chat_id = call.message.chat.id
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

        df = pd.read_csv("Rehabilitation_imputed_whole_grain_timeframe.csv")
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

        result_text = f"✅ *Final Functional Muscle Change Index after amputation:* `{round(pred, 2)}` index units\n\n"

        is_right_shoulder_amp = 'amp_shoulder_r' in selected_amps
        if is_right_shoulder_amp:
            result_text += "⚠️ *Attention:* UAMC clinical anthropometry (Frisancho) is not possible due to the amputation of the right arm.\n\n"
        else:
            if is_upper:
                result_text += "⚠️ *Attention:* Since the amputation involves the upper limb, the UAMC metric may not be fully representative.\n"

        markup = types.InlineKeyboardMarkup()
        markup.add(types.InlineKeyboardButton("🔄 Start Over", callback_data="restart"))

        bot.send_message(chat_id, result_text, parse_mode="Markdown", reply_markup=markup)

    except Exception as e:
        msg = bot.send_message(chat_id, "Enter a valid number (number of days):")
        bot.register_next_step_handler(msg, process_amp_days_and_predict)

@bot.callback_query_handler(func=lambda call: call.data == "restart")
def handle_restart_click(call):
    chat_id = call.message.chat.id
    if chat_id in patient_symptoms:
        del patient_symptoms[chat_id]

    try:
        bot.answer_callback_query(call.id)
    except ApiTelegramException:
        pass

    start_message(call.message)

if __name__ == '__main__':
    bot.polling(none_stop=True)
