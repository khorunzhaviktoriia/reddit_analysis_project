"""
pip install pyreadstat
pip install pandas
python combine_pew_surveys.py --zips ./pew_zips --out ./pew_out
--zips  папка з оригінальними .zip-файлами, як вони завантажені з сайту Pew
--out   папка для аутпуту
"""

import argparse
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

CONFIG = [
    dict(wave_id="Feb17", zip="Feb17-public.zip", family="phone_rdd",
         id="psraid", weight="weight", approval="q1", strength="q1a",
         date="int_date", date_fmt="mddyy", party="partysum", ideo="ideo",
         gender="sex", age="age", educ="educ2"),
    dict(wave_id="Apr17", zip="Apr17-public-4_3-update.zip", family="phone_rdd",
         id="psraid", weight="weight", approval="q1", strength="q1a",
         date="int_date", date_fmt="mddyy", party="partysum", ideo="ideo",
         gender="sex", age="age", educ="educ2"),
    dict(wave_id="Typology17", zip="Typology-17.zip", family="phone_rdd",
         id="mergeid", weight="weight", approval="qa1", strength="qa1a", keep_if_notna="qa1",
         date="int_date", date_fmt="mddyy", party="partysum", ideo="ideo",
         gender="sex", age="age", educ="educ2"),
    dict(wave_id="Jan18", zip="Jan18.zip", family="phone_rdd",
         id="respid", weight="weight", approval="q2", strength="q2a",
         date="int_date", date_fmt="yymmdd", party="partysum", ideo="ideo",
         gender="sex", age="age", educ="educ"),
    dict(wave_id="Mar18", zip="March18.zip", family="phone_rdd",
         id="masterid", weight="weight", approval="q2", strength=None,
         date="ftcalldt", date_fmt="date", party="partysum", ideo="ideo",
         gender="sex", age="age", educ="educ"),
    dict(wave_id="May18", zip="May18.zip", family="phone_rdd",
         id="respid", weight="weight", approval="q2", strength="q2a",
         date="int_date", date_fmt="yymmdd", party="partysum", ideo="ideo",
         gender="sex", age="age", educ="educ"),
    dict(wave_id="Jun18", zip="June18.zip", family="phone_rdd",
         id="respid", weight="weight", approval="q2", strength=None,
         date="int_date", date_fmt="yymmdd", party="partysum", ideo="ideo",
         gender="sex", age="age", educ="educ"),
    dict(wave_id="W161", zip="W161_Feb25.zip", family="atp",
         id="QKEY", weight="WEIGHT_W161", approval="POL1DT_W161", strength="POL1DTSTR_W161",
         date="INTERVIEW_START_W161", date_fmt="atp", party="F_PARTYSUM_FINAL", ideo="F_IDEO",
         gender="F_GENDER", agecat="F_AGECAT", educcat="F_EDUCCAT", mode="SVYMODE_W161"),
    dict(wave_id="W167", zip="W167_Apr25.zip", family="atp",
         id="QKEY", weight="WEIGHT_W167", approval="POL1DT_W167", strength="POL1DTSTR_W167",
         date="INTERVIEW_START_W167", date_fmt="atp", party="F_PARTYSUM_FINAL", ideo="F_IDEO",
         gender="F_GENDER", agecat="F_AGECAT", educcat="F_EDUCCAT", mode="SVYMODE_W167"),
]


#  Код "не знаю/відмова": 9 у телефонних опитуваннях, 99 в ATP.

APPROVAL = {1: "approve", 2: "disapprove", 9: "dk", 99: "dk"}
STRENGTH = {1: "very_strongly", 2: "not_so_strongly", 9: "dk", 99: "dk"}
PARTY = {1: "rep_lean", 2: "dem_lean", 9: "other_dk"}   # rep/dem разом із тими, хто "схиляється"
IDEO = {1: "very_conservative", 2: "conservative", 3: "moderate", 4: "liberal", 5: "very_liberal"}
GENDER_PHONE = {1: "man", 2: "woman"}
GENDER_ATP = {1: "man", 2: "woman", 3: "other"}
AGECAT = {1: "18-29", 2: "30-49", 3: "50-64", 4: "65+"}
EDUC_ATP = {1: "college_grad_plus", 2: "some_college", 3: "hs_or_less"}
ATP_MODE = {1: "atp_web", 2: "atp_phone"}


def educ_phone(code):
    if pd.isna(code) or code >= 9:   # 9 = відмова
        return np.nan
    if code <= 3:                     # до випускника школи включно
        return "hs_or_less"
    if code <= 5:                     # коледж без диплома бакалавра / associate degree
        return "some_college"
    return "college_grad_plus"        # бакалавр і вище


def age_group(age):
    if pd.isna(age) or age >= 99:  # 99 = відмова
        return np.nan
    if age <= 29:
        return "18-29"
    if age <= 49:
        return "30-49"
    if age <= 64:
        return "50-64"
    return "65+"


def parse_dates(s, fmt):
    if fmt == "mddyy":
        txt = s.dropna().astype("int64").astype(str).str.zfill(6)
        out = pd.to_datetime(txt, format="%m%d%y", errors="coerce")
    elif fmt == "yymmdd":
        txt = s.dropna().astype("int64").astype(str).str.zfill(6)
        out = pd.to_datetime(txt, format="%y%m%d", errors="coerce")
    elif fmt == "date":
        out = pd.to_datetime(s.dropna())
    elif fmt == "atp":
        out = pd.to_datetime(s.dropna(), format="%d-%b-%Y %H:%M:%S", errors="coerce")
    else:
        raise ValueError(fmt)
    return out.reindex(s.index).dt.normalize()


def find_zip(zips_dir: Path, name: str) -> Path:
    exact = zips_dir / name
    if exact.exists():
        return exact
    norm = lambda t: "".join(ch for ch in t.lower() if ch.isalnum())
    hits = [p for p in zips_dir.glob("*.zip") if norm(p.name) == norm(name)]
    if len(hits) != 1:
        raise FileNotFoundError(f"Cannot find {name} in {zips_dir} (candidates: {[p.name for p in hits]})")
    return hits[0]


def clean_id(s):
    # У більшості файлів ID це числа з крапкою (123.0 -> '123'), але в March18 це рядки типу '00140838F
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("int64").astype(str)
    return s.astype(str).str.strip()


def read_sav_from_zip(zip_path: Path, tmpdir: Path):
    with zipfile.ZipFile(zip_path) as z:
        sav_members = [m for m in z.namelist() if m.lower().endswith(".sav")]
        assert len(sav_members) == 1, f"{zip_path.name}: expected exactly one .sav, got {sav_members}"
        extracted = z.extract(sav_members[0], tmpdir)

    return pyreadstat.read_sav(extracted, apply_value_formats=False)


def harmonize_one(cfg, df):
    out = pd.DataFrame(index=df.index)
    out["wave_id"] = cfg["wave_id"]
    out["source_family"] = cfg["family"]
    out["orig_id"] = clean_id(df[cfg["id"]])
    out["interview_date"] = parse_dates(df[cfg["date"]], cfg["date_fmt"])

    if cfg["family"] == "atp":
        out["mode"] = df[cfg["mode"]].map(ATP_MODE)
    else:
        out["mode"] = "phone_rdd"

    out["trump_approval"] = df[cfg["approval"]].map(APPROVAL)  # NaN = респонденту це питання не ставили
    out["approval_strength"] = df[cfg["strength"]].map(STRENGTH) if cfg.get("strength") else np.nan
    out["party_lean"] = df[cfg["party"]].map(PARTY)
    out["ideology"] = df[cfg["ideo"]].map(IDEO)  # "не знаю/відмова" стає NaN

    if cfg["family"] == "phone_rdd":
        out["gender"] = df[cfg["gender"]].map(GENDER_PHONE)
        out["age"] = df[cfg["age"]].where(df[cfg["age"]] < 99)  # точний вік є лише в телефонних файлах
        out["age_group"] = out["age"].map(age_group)
        out["education"] = df[cfg["educ"]].map(educ_phone)
    else:
        out["gender"] = df[cfg["gender"]].map(GENDER_ATP)
        out["age"] = np.nan  # у публічних файлах ATP є лише вікові категорії
        out["age_group"] = df[cfg["agecat"]].map(AGECAT)
        out["education"] = df[cfg["educcat"]].map(EDUC_ATP)

    out["weight_raw"] = df[cfg["weight"]]
    out["weight_norm"] = out["weight_raw"] / out["weight_raw"].mean()
    return out


def wave_qa(resp):
    rows = []
    for wave, g in resp.groupby("wave_id"):
        asked = g[g["trump_approval"].notna()]
        w = asked["weight_raw"]
        rows.append({
            "wave_id": wave,
            "field_start": g["interview_date"].min().date(),
            "field_end": g["interview_date"].max().date(),
            "n_respondents": len(g),
            "n_asked_approval": len(asked),
            "approve_pct": 100 * w[asked["trump_approval"] == "approve"].sum() / w.sum(),
        })
    return pd.DataFrame(rows).sort_values("field_start").reset_index(drop=True)


def main(zips_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, crosswalk = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for cfg in CONFIG:
            zpath = find_zip(zips_dir, cfg["zip"])
            df, meta = read_sav_from_zip(zpath, Path(tmp))

            if cfg.get("keep_if_notna"):
                n_before = len(df)
                df = df[df[cfg["keep_if_notna"]].notna()].copy()
                print(f"[filter] {cfg['wave_id']}: kept {len(df)}/{n_before} respondents who were asked '{cfg['keep_if_notna']}'")

            missing = [v for k, v in cfg.items()
                       if k in {"id", "weight", "approval", "strength", "date", "party", "ideo",
                                "gender", "age", "educ", "agecat", "educcat", "mode"}
                       and v and v not in df.columns]
            assert not missing, f"{cfg['wave_id']}: variables not found in file: {missing}"
            harmonized = harmonize_one(cfg, df)
            frames.append(harmonized)
            # Записуємо в crosswalk, яка сира змінна (і з яким текстовим описом) стала якою роллю
            labels = dict(zip(meta.column_names, meta.column_labels))
            for role in ["id", "weight", "approval", "strength", "date", "party", "ideo",
                         "gender", "age", "agecat", "educ", "educcat", "mode"]:
                if cfg.get(role):
                    crosswalk.append(dict(wave_id=cfg["wave_id"], role=role, raw_variable=cfg[role],
                                          raw_label=(labels.get(cfg[role]) or "")[:150]))
            print(f"[ok] {cfg['wave_id']:<11} raw shape {df.shape} -> kept {harmonized.shape[1]} harmonized columns")

    resp = pd.concat(frames, ignore_index=True)

    assert not resp.duplicated(["wave_id", "orig_id"]).any(), "duplicate respondent IDs inside a wave"
    assert resp["interview_date"].notna().all(), "unparsed interview dates"
    assert resp["weight_raw"].notna().all() and (resp["weight_raw"] > 0).all(), "bad weights"

    qa = wave_qa(resp)

    span = (pd.to_datetime(qa["field_end"]) - pd.to_datetime(qa["field_start"])).dt.days
    assert (span <= 21).all(), f"suspiciously long fieldwork (mixed phases?): {qa.loc[span > 21, 'wave_id'].tolist()}"

    resp.to_csv(out_dir / "pew_surveys.csv", index=False)
    pd.DataFrame(crosswalk).to_csv(out_dir / "pew_variable_crosswalk.csv", index=False)

    print("\n=== QA REPORT ===")
    print("stacked table:", resp.shape)
    print("\nrespondents per wave:\n", resp["wave_id"].value_counts().sort_index().to_string())
    print("\nshare of missing values per harmonized column (%):")
    print((resp.isna().mean() * 100).round(1).to_string())
    print("\nQA by survey wave (weighted approval, %):")
    print(qa.round(1).to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("./pew_out"))
    args = ap.parse_args()
    main(args.zips, args.out)