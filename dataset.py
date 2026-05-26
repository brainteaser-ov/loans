from pathlib import Path
import pandas as pd


BORROWINGS = "borrowings.csv"
FORMS = "forms.csv"
LANGUAGES = "languages.csv"
PARAMETERS = "parameters.csv"
OUT_XLSX = Path("dataset.xlsx")

borrowings = pd.read_csv(BORROWINGS, dtype=str).fillna("")
forms = pd.read_csv(FORMS, dtype=str).fillna("")
languages = pd.read_csv(LANGUAGES, dtype=str).fillna("")
parameters = pd.read_csv(PARAMETERS, dtype=str).fillna("")


def get_value(row, *names):
    for name in names:
        if name in row:
            return row[name]
    return ""


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
unmatched = []

for _, b in borrowings.iterrows():
    try:
        lang_id, parameter_id, occurrence = parse_target_form_id(b["Target_Form_ID"])

        if lang_id not in lang_by_id:
            raise KeyError(f"Language not found: {lang_id}")

        if parameter_id not in param_by_id:
            raise KeyError(f"Parameter not found: {parameter_id}")

        lang = lang_by_id[lang_id]
        parameter = param_by_id[parameter_id]
        target_wold_id = str(lang["WOLD_ID"])

        group_key = (target_wold_id, parameter_id)

        if group_key not in form_groups:
            raise KeyError(f"Form group not found: {group_key}")

        candidates = form_groups[group_key]

        if occurrence < 1 or occurrence > len(candidates):
            raise IndexError(
                f"Occurrence {occurrence} out of range for {group_key}; "
                f"available forms: {len(candidates)}"
            )

        target = candidates[occurrence - 1]

        records.append({
            "ID": b["ID"],
            "Meaning": get_value(parameter, "Name"),
            "SemanticField": get_value(parameter, "Semantic_field"),
            "SemanticCategory": get_value(parameter, "Semantic_category"),
            "Target_Form": get_value(target, "Form"),
            "Target_Language_Name": get_value(lang, "Name"),
            "Donor": get_value(b, "Source_word"),
            "Source_Language": get_value(b, "Source_languoid"),
            "Source_Glottocode": get_value(b, "Source_languoid_glottocode"),
            "Source_Meaning": get_value(b, "Source_meaning"),
            "Source_Relation": get_value(b, "Source_relation"),
            "Source_Certain": get_value(b, "Source_certain"),
            "Target_Form_ID": get_value(b, "Target_Form_ID"),
            "Target_Form_Row_ID": get_value(target, "ID"),
            "Target_WOLD_ID": get_value(lang, "WOLD_ID"),
            "Target_Language_ID_CLDF": get_value(target, "Language_ID"),
            "Target_Glottocode": get_value(lang, "Glottocode"),
            "Target_Macroarea": get_value(lang, "Macroarea"),
            "Target_Family": get_value(lang, "Family"),
            "Parameter_ID": parameter_id,
            "Concepticon_ID": get_value(parameter, "Concepticon_ID"),
            "Concepticon_Gloss": get_value(parameter, "Concepticon_Gloss"),
            "Target_Segments": get_value(target, "Segments"),
            "Target_Borrowed": get_value(target, "Borrowed"),
            "Target_Borrowed_Score": get_value(target, "Borrowed_score", "BorrowedScore"),
            "Target_Age_Score": get_value(target, "Age_score", "AgeScore"),
            "Target_Simplicity_Score": get_value(target, "Simplicity_score", "SimplicityScore"),
            "Source_Form_ID": get_value(b, "Source_Form_ID"),
            "Borrowing_Comment": get_value(b, "Comment"),
            "Borrowing_Source": get_value(b, "Source"),
            "Join_Method": "Target_Form_ID parsed as language + parameter + form number",
            "Target_Form_Occurrence": occurrence,
            "label_loan": 1 if str(get_value(b, "Source_word")).strip() else "",
        })

    except Exception as error:
        unmatched.append({
            "Borrowing_ID": get_value(b, "ID"),
            "Target_Form_ID": get_value(b, "Target_Form_ID"),
            "Error": str(error),
        })


dataset = pd.DataFrame.from_records(records)
unmatched_df = pd.DataFrame.from_records(unmatched)

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    dataset.to_excel(writer, index=False, sheet_name="dataset")

    summary = pd.DataFrame([
        ["Строк в borrowings.csv", len(borrowings)],
        ["Строк в forms.csv", len(forms)],
        ["Строк в languages.csv", len(languages)],
        ["Строк в parameters.csv", len(parameters)],
        ["Строк итогового датасета", len(dataset)],
        ["Несвязанных строк", len(unmatched_df)],
        ["Столбцов итогового датасета", dataset.shape[1]],
        [
            "Target_Borrowed_Score > 0",
            (pd.to_numeric(dataset["Target_Borrowed_Score"], errors="coerce") > 0).sum()
            if "Target_Borrowed_Score" in dataset.columns else 0
        ],
        [
            "Target_Borrowed_Score = 1",
            (pd.to_numeric(dataset["Target_Borrowed_Score"], errors="coerce") == 1).sum()
            if "Target_Borrowed_Score" in dataset.columns else 0
        ],
    ], columns=["Показатель", "Значение"])

    summary.to_excel(writer, index=False, sheet_name="summary")

    if len(unmatched_df) > 0:
        unmatched_df.to_excel(writer, index=False, sheet_name="unmatched")

print(f"Готово: {OUT_XLSX}")
print(f"Строк в датасете: {len(dataset)}")
print(f"Несвязанных строк: {len(unmatched_df)}")
