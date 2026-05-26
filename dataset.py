from pathlib import Path
import pandas as pd

CLDF_DIR = Path("data/cldf")
BORROWINGS = CLDF_DIR / "borrowings.csv"
FORMS = CLDF_DIR / "forms.csv"
LANGUAGES = CLDF_DIR / "languages.csv"
PARAMETERS = CLDF_DIR / "parameters.csv"
OUT_XLSX = Path("wold_borrowings_dataset.xlsx")

borrowings = pd.read_csv(BORROWINGS, dtype=str).fillna("")
forms = pd.read_csv(FORMS, dtype=str).fillna("")
languages = pd.read_csv(LANGUAGES, dtype=str).fillna("")
parameters = pd.read_csv(PARAMETERS, dtype=str).fillna("")

lang_by_id = languages.set_index("ID").to_dict("index")
param_by_id = parameters.set_index("ID").to_dict("index")
lang_names = sorted(lang_by_id.keys(), key=len, reverse=True)

form_groups = {}
for _, row in forms.iterrows():
    key = (str(row["Language_ID"]), str(row["Parameter_ID"]))
    form_groups.setdefault(key, []).append(row.to_dict())


def parse_target_form_id(value):
    for lang_id in lang_names:
        prefix = f"{lang_id}-"
        if value.startswith(prefix):
            rest = value[len(prefix):]
            parameter_id, occurrence = rest.rsplit("-", 1)
            return lang_id, parameter_id, int(occurrence)
    raise ValueError(f"Cannot parse Target_Form_ID: {value}")

records = []
for _, b in borrowings.iterrows():
    lang_id, parameter_id, occurrence = parse_target_form_id(b["Target_Form_ID"])
    lang = lang_by_id[lang_id]
    parameter = param_by_id[parameter_id]
    target_wold_id = str(lang["WOLD_ID"])
    target = form_groups[(target_wold_id, parameter_id)][occurrence - 1]

    records.append({
        "ID": b["ID"],
        "Meaning": parameter["Name"],
        "SemanticField": parameter["Semantic_field"],
        "SemanticCategory": parameter["Semantic_category"],
        "Target_Form": target["Form"],
        "Target_Language_Name": lang["Name"],
        "Donor": b["Source_word"],
        "Source_Language": b["Source_languoid"],
        "Source_Glottocode": b["Source_languoid_glottocode"],
        "Source_Meaning": b["Source_meaning"],
        "Source_Relation": b["Source_relation"],
        "Source_Certain": b["Source_certain"],
        "Target_Form_ID": b["Target_Form_ID"],
        "Target_Form_Row_ID": target["ID"],
        "Target_WOLD_ID": lang["WOLD_ID"],
        "Target_Language_ID_CLDF": target["Language_ID"],
        "Target_Glottocode": lang["Glottocode"],
        "Target_Macroarea": lang["Macroarea"],
        "Target_Family": lang["Family"],
        "Parameter_ID": parameter["ID"],
        "Concepticon_ID": parameter["Concepticon_ID"],
        "Concepticon_Gloss": parameter["Concepticon_Gloss"],
        "Target_Segments": target["Segments"],
        "Target_Borrowed": target["Borrowed"],
        "Target_Borrowed_Score": target["BorrowedScore"],
        "Target_Age_Score": target["AgeScore"],
        "Target_Simplicity_Score": target["SimplicityScore"],
        "Source_Form_ID": b["Source_Form_ID"],
        "Borrowing_Comment": b["Comment"],
        "Borrowing_Source": b["Source"],
        "Join_Method": "Target_Form_ID parsed as language + parameter + form number",
        "Target_Form_Occurrence": occurrence,
        "label_loan": 1 if str(b["Source_word"]).strip() else "",
    })

dataset = pd.DataFrame.from_records(records)

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    dataset.to_excel(writer, index=False, sheet_name="dataset")
    summary = pd.DataFrame([
        ["Строк в borrowings.csv", len(borrowings)],
        ["Строк в forms.csv", len(forms)],
        ["Строк в languages.csv", len(languages)],
        ["Строк в parameters.csv", len(parameters)],
        ["Строк итогового датасета", len(dataset)],
        ["Столбцов итогового датасета", dataset.shape[1]],
        ["Target_Borrowed_Score > 0", (pd.to_numeric(dataset["Target_Borrowed_Score"], errors="coerce") > 0).sum()],
        ["Target_Borrowed_Score = 1", (pd.to_numeric(dataset["Target_Borrowed_Score"], errors="coerce") == 1).sum()],
    ], columns=["Показатель", "Значение"])
    summary.to_excel(writer, index=False, sheet_name="summary")
