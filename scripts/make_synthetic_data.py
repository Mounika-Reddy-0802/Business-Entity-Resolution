"""Synthetic stand-in for the organiser dataset, with the exact file names and columns.

Only used while the real files are missing from data/raw/dataset (see docs/problems.md). It
reproduces the noise patterns listed in the problem statement: legal-suffix changes, & vs and,
typos, word reordering, truncated trade names, address abbreviations, missing postal codes,
landmark references, reordered components and transliteration variants. Train has US and India;
test adds France. Hard negatives (same brand in another city, same street, similar names) are
included so precision is not trivial.

    python scripts/make_synthetic_data.py [--scale 1.0]
"""
import argparse
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "dataset"

US = {
    "brand": ["Acme", "Blue River", "Summit", "Pioneer", "Golden Gate", "Liberty", "Evergreen",
              "Redwood", "Silver Oak", "Northstar", "Harbor", "Maple Leaf", "Keystone", "Lakeside",
              "Sunrise", "Patriot", "Frontier", "Cascade", "Bright Path", "Iron Horse", "Crescent",
              "Prairie", "Granite", "Coastal", "Heritage", "Magnolia", "Beacon", "Sterling"],
    "person": ["Johnson", "Smith", "Miller", "Garcia", "Davis", "Wilson", "Anderson", "Thompson",
               "Martinez", "Robinson", "Clark", "Lewis", "Walker", "Young", "Allen", "Wright",
               "Hernandez", "King", "Scott", "Green", "Baker", "Adams", "Nelson", "Carter"],
    "kind": ["Dental Care", "Auto Repair", "Consulting", "Plumbing", "Bakery", "Pharmacy",
             "Law Group", "Realty", "Insurance Agency", "Hardware", "Fitness Center", "Pizza",
             "Coffee House", "Landscaping", "Electric", "Veterinary Clinic", "Printing",
             "Roofing", "Tax Services", "Family Medicine", "Car Wash", "Florist", "Grill"],
    "suffix": [("Inc", "Incorporated"), ("LLC", "L.L.C."), ("Corp", "Corporation"),
               ("Co", "Company"), ("Ltd", "Limited"), ("", "")],
    "street": ["Main", "Oak", "Maple", "Washington", "Park", "Elm", "Lake", "Hill", "Cedar",
               "Pine", "Sunset", "Lincoln", "Jackson", "Church", "Highland", "Madison", "Franklin"],
    "stype": [("St", "Street"), ("Ave", "Avenue"), ("Rd", "Road"), ("Blvd", "Boulevard"),
              ("Dr", "Drive"), ("Ln", "Lane")],
    "city": [("Austin", "TX", "787"), ("Denver", "CO", "802"), ("Seattle", "WA", "981"),
             ("Boston", "MA", "021"), ("Phoenix", "AZ", "850"), ("Columbus", "OH", "432"),
             ("Portland", "OR", "972"), ("Atlanta", "GA", "303"), ("Chicago", "IL", "606"),
             ("Nashville", "TN", "372"), ("Raleigh", "NC", "276"), ("Tampa", "FL", "336")],
}

IN = {
    "brand": ["Sri Lakshmi", "Balaji", "Ganesh", "Sai Krishna", "Venkateswara", "Om Sai",
              "Shree Ram", "Durga", "Annapurna", "Maruti", "Vinayaka", "Hanuman", "Sri Sai",
              "Mahalakshmi", "Tirumala", "Jai Hind", "Navaratna", "Kaveri", "Ganga", "Vijaya"],
    "person": ["Sharma", "Reddy", "Patel", "Gupta", "Agarwal", "Iyer", "Nair", "Rao", "Singh",
               "Kumar", "Jain", "Mehta", "Chowdary", "Verma", "Naidu", "Pillai", "Das", "Bose"],
    "kind": ["Traders", "Electronics", "Medical Stores", "Enterprises", "Textiles",
             "Jewellers", "Hardwares", "Sweets", "Tiffin Centre", "Mobile Shop", "Agencies",
             "Automobiles", "Kirana Store", "Hospital", "Opticals", "Book Depot", "Steels",
             "Tours and Travels", "Furniture", "Bakery"],
    "suffix": [("Pvt Ltd", "Private Limited"), ("LLP", "L.L.P."), ("& Co", "and Company"),
               ("Ltd", "Limited"), ("", ""), ("", "")],
    "area": ["Gandhi Nagar", "Ashok Nagar", "Nehru Colony", "Jubilee Hills", "MG Road",
             "Anna Nagar", "Rajaji Nagar", "Banjara Hills", "Koramangala", "Indira Nagar",
             "Shivaji Nagar", "Laxmi Nagar", "Kamla Nagar", "Station Road", "Main Bazaar"],
    "landmark": ["SBI ATM", "Bus Stand", "Railway Station", "Govt Hospital", "Hanuman Temple",
                 "Post Office", "Clock Tower", "Police Station", "Big Bazaar", "HDFC Bank"],
    "city": [("Hyderabad", "Telangana", "500", "Hyderabad"),
             ("Bengaluru", "Karnataka", "560", "Bangalore"),
             ("Mumbai", "Maharashtra", "400", "Bombay"), ("Chennai", "Tamil Nadu", "600", "Madras"),
             ("Pune", "Maharashtra", "411", "Poona"), ("Kolkata", "West Bengal", "700", "Calcutta"),
             ("Vijayawada", "Andhra Pradesh", "520", "Bezawada"),
             ("Delhi", "Delhi", "110", "New Delhi"), ("Jaipur", "Rajasthan", "302", "Jaipur")],
}

FR = {
    "brand": ["Le Petit", "La Belle", "Au Bon", "L'Atelier", "Le Grand", "Les Délices",
              "Maison", "Le Comptoir", "La Table", "L'Étoile", "Le Relais", "Aux Saveurs"],
    "person": ["Dupont", "Martin", "Bernard", "Dubois", "Durand", "Lefèvre", "Moreau", "Laurent",
               "Girard", "Rousseau", "Fontaine", "Chevalier", "Garnier", "Faure", "Mercier"],
    "kind": ["Boulangerie", "Pharmacie", "Boucherie", "Garage", "Café", "Coiffure", "Fleuriste",
             "Librairie", "Pâtisserie", "Cabinet Dentaire", "Brasserie", "Immobilier",
             "Plomberie", "Tabac Presse", "Optique", "Épicerie"],
    "suffix": [("SARL", "S.A.R.L."), ("SAS", "S.A.S."), ("SA", "S.A."), ("EURL", "E.U.R.L."),
               ("", ""), ("", "")],
    "street": ["de la République", "Victor Hugo", "du Général de Gaulle", "Jean Jaurès",
               "de la Gare", "Pasteur", "des Écoles", "de Paris", "du Moulin", "Saint-Michel",
               "de l'Église", "Nationale", "Gambetta", "Voltaire"],
    "stype": [("rue", "rue"), ("av.", "avenue"), ("bd", "boulevard"), ("pl.", "place"),
              ("ch.", "chemin")],
    "city": [("Paris", "75"), ("Lyon", "69"), ("Marseille", "13"), ("Toulouse", "31"),
             ("Nantes", "44"), ("Bordeaux", "33"), ("Lille", "59"), ("Strasbourg", "67")],
}


def typo(s, rng):
    """One random character edit inside a word longer than three letters."""
    idx = [i for i, c in enumerate(s) if c.isalpha()]
    if len(idx) < 4:
        return s
    i = rng.choice(idx[1:-1])
    op = rng.random()
    if op < 0.33:
        return s[:i] + s[i + 1:]
    if op < 0.66 and i + 1 < len(s):
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + rng.choice("aeiorstn") + s[i:]


def strip_accents(s):
    table = str.maketrans("éèêëàâäîïôöùûüçÉÈÊÀÂÎÔÙÛÇ", "eeeeaaaiioouuucEEEAAIOUUC")
    return s.translate(table)


class Gen:
    def __init__(self, rng):
        self.rng = rng

    # ---------- canonical entities ----------
    def entity(self, country, brand_name=None, city=None):
        """A canonical business: dict with name parts and address parts."""
        rng = self.rng
        d = {"US": US, "India": IN, "France": FR}[country]
        if brand_name is None:
            style = rng.random()
            if country == "France":
                if style < 0.5:
                    core = f"{rng.choice(d['kind'])} {rng.choice(d['person'])}"
                else:
                    core = f"{rng.choice(d['brand'])} {rng.choice(d['kind'])}"
            elif style < 0.45:
                core = f"{rng.choice(d['brand'])} {rng.choice(d['kind'])}"
            elif style < 0.8:
                core = f"{rng.choice(d['person'])} {rng.choice(d['kind'])}"
            else:
                a, b = rng.sample(d["person"], 2)
                core = f"{a} & {b} {rng.choice(d['kind'])}"
        else:
            core = brand_name
        suffix = rng.choice(d["suffix"])
        num = str(rng.randint(1, 9999)) if country != "France" else str(rng.randint(1, 180))
        if country == "US":
            c = city or rng.choice(d["city"])
            addr = {"num": num, "street": rng.choice(d["street"]), "stype": rng.choice(d["stype"]),
                    "city": c, "zip": c[2] + f"{rng.randint(0, 99):02d}",
                    "unit": f"Suite {rng.randint(100, 900)}" if rng.random() < 0.2 else ""}
        elif country == "India":
            c = city or rng.choice(d["city"])
            addr = {"num": f"{rng.randint(1, 40)}-{rng.randint(1, 300)}" if rng.random() < 0.4 else num[:3],
                    "area": rng.choice(d["area"]), "landmark": rng.choice(d["landmark"]),
                    "city": c, "pin": c[2] + f"{rng.randint(0, 99):03d}",
                    "floor": rng.choice(["", "1st Floor", "Ground Floor", "2nd Floor"])}
        else:
            c = city or rng.choice(d["city"])
            addr = {"num": num, "street": rng.choice(d["street"]), "stype": rng.choice(d["stype"]),
                    "city": c, "cp": c[1] + f"{rng.randint(0, 20):03d}"}
        return {"country": country, "core": core, "suffix": suffix, "addr": addr}

    # ---------- rendering with noise ----------
    def render_name(self, e, noise):
        rng = self.rng
        core, (short, long_) = e["core"], e["suffix"]
        words = core.split()
        if noise and rng.random() < 0.15 and len(words) > 2:
            words = words[:2]                              # trade / DBA name
        if noise and rng.random() < 0.12 and len(words) > 1:
            i = rng.randrange(len(words) - 1)
            words[i], words[i + 1] = words[i + 1], words[i]  # word transposition
        name = " ".join(words)
        if noise and rng.random() < 0.25:
            name = name.replace("&", "and") if "&" in name else name.replace(" and ", " & ")
        if short:
            r = rng.random() if noise else 0.0
            if r < 0.5:
                name = f"{name} {short}"
            elif r < 0.75:
                name = f"{name} {long_}"
            elif r < 0.85:
                name = f"{name}, {short}."
        if noise and rng.random() < 0.2:
            name = typo(name, rng)
        if noise and rng.random() < 0.15:
            name = name.upper()
        if noise and rng.random() < 0.3:
            name = strip_accents(name)
        if noise and rng.random() < 0.1:
            name = name.replace(".", "").replace("'", " ")
        return name

    def render_addr(self, e, noise):
        rng = self.rng
        a, country = e["addr"], e["country"]
        ab = lambda pair: pair[0] if (noise and rng.random() < 0.5) else pair[1]
        drop = lambda p: noise and rng.random() < p
        if country == "US":
            city, state, _ = a["city"]
            num = a["num"] if not drop(0.05) else ""
            if noise and rng.random() < 0.1:
                num = f"#{num}" if num else num
            parts = [f"{num} {a['street']} {ab(a['stype'])}".strip()]
            if a["unit"] and not drop(0.4):
                parts.append(a["unit"] if not noise or rng.random() < 0.5 else a["unit"].replace("Suite", "Ste"))
            if not drop(0.1):
                parts.append(city)
            tail = state if not drop(0.15) else ""
            if not drop(0.25):
                tail = f"{tail} {a['zip']}".strip()
            if tail:
                parts.append(tail)
        elif country == "India":
            city, state, _, alt = a["city"]
            num = a["num"]
            if noise and rng.random() < 0.3:
                num = rng.choice([f"No. {num}", f"Shop No {num}", f"H.No {num}", f"#{num}"])
            parts = [num]
            if a["floor"] and not drop(0.5):
                parts.append(a["floor"])
            area = a["area"]
            if noise and rng.random() < 0.3:
                area = area.replace("Nagar", rng.choice(["Nagr", "Ngr", "nagar"])).replace(
                    "Road", "Rd").replace("Colony", "Col")
            parts.append(area)
            if not drop(0.45):
                lm = a["landmark"]
                parts.append(f"{rng.choice(['Near', 'Nr', 'Opp', 'Opposite', 'Beside']) if noise else 'Near'} {lm}")
            parts.append(alt if (noise and rng.random() < 0.3) else city)
            if not drop(0.4):
                parts.append(state)
            if not drop(0.35):
                parts[-1] = f"{parts[-1]} - {a['pin']}" if rng.random() < 0.5 else f"{parts[-1]} {a['pin']}"
            if noise and rng.random() < 0.15:
                rng.shuffle(parts)
        else:
            city, _ = a["city"]
            stype = ab(a["stype"])
            parts = [f"{a['num']}{',' if noise and rng.random() < 0.3 else ''} {stype} {a['street']}"]
            if not drop(0.25):
                parts.append(f"{a['cp']} {city}")
            else:
                parts.append(city.upper() if noise and rng.random() < 0.5 else city)
            if noise and rng.random() < 0.3:
                parts = [strip_accents(p) for p in parts]
        s = ", ".join(p for p in parts if p)
        if noise and rng.random() < 0.15:
            s = typo(s, rng)
        if noise and rng.random() < 0.1:
            s = s.upper()
        return s


def build_split(prefix, countries, n_s1, seed, id_start):
    """Generate S1/S2/S3 frames and ground truth for one split."""
    rng = random.Random(seed)
    g = Gen(rng)
    s1, s2, s3, gt = [], [], [], []
    counter = {"S1": id_start, "S2": id_start, "S3": id_start}

    def new_id(src):
        counter[src] += 1
        return f"{src}-{counter[src]:06d}"

    def add(src, e, noise=True):
        rec = {"entity_id": None, "business_name": g.render_name(e, noise),
               "business_address": g.render_addr(e, noise), "country": e["country"]}
        (s2 if src == "S2" else s3).append(rec)
        return id(rec)

    canon = []
    for country, share in countries:
        n = int(n_s1 * share)
        chains = [g.entity(country)["core"] for _ in range(max(3, n // 60))]
        for _ in range(n):
            if rng.random() < 0.12:
                canon.append(g.entity(country, brand_name=rng.choice(chains)))   # chain branch
            else:
                canon.append(g.entity(country))
    rng.shuffle(canon)
    for e in canon:
        sid = new_id("S1")
        s1.append({"entity_id": sid, "business_name": g.render_name(e, rng.random() < 0.3),
                   "business_address": g.render_addr(e, rng.random() < 0.3), "country": e["country"]})
        k = rng.choices([0, 1, 2, 3, 4, 5], weights=[32, 34, 18, 9, 5, 2])[0]
        matches = [add("S2" if rng.random() < 0.6 else "S3", e) for _ in range(k)]
        gt.append({"source1_entity_id": sid, "matches": matches})
        if rng.random() < 0.08:                                   # same name, other city
            twin = g.entity(e["country"], brand_name=e["core"])
            add("S2" if rng.random() < 0.5 else "S3", twin)
        if rng.random() < 0.05:                                   # same address, other business
            other = g.entity(e["country"], city=e["addr"]["city"])
            other["addr"] = dict(e["addr"])
            add("S2" if rng.random() < 0.5 else "S3", other)
    for country, share in countries:                              # records with no S1 entity
        for _ in range(int(n_s1 * share * 0.35)):
            add("S2" if rng.random() < 0.55 else "S3", g.entity(country))
    real_id = {}
    for src, rows in (("S2", s2), ("S3", s3)):              # ids assigned after shuffling so
        rng.shuffle(rows)                                    # they carry no ordering signal
        for rec in rows:
            real_id[id(rec)] = new_id(src)
            rec["entity_id"] = real_id[id(rec)]
    gt = [{"source1_entity_id": r["source1_entity_id"],
           "matched_entity_ids": ",".join(real_id[m] for m in r["matches"])} for r in gt]
    return s1, s2, s3, gt


def write(prefix, s1, s2, s3, gt=None):
    d = OUT / prefix
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in (("source1", s1), ("source2", s2), ("source3", s3)):
        pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"]
                     ).to_csv(d / f"{prefix}_{name}.tsv", sep="\t", index=False)
    if gt is not None:
        pd.DataFrame(gt, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
            d / f"{prefix}_ground_truth.tsv", sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    s1, s2, s3, gt = build_split("train", [("US", 0.5), ("India", 0.5)], int(10000 * args.scale), 42, 0)
    write("train", s1, s2, s3, gt)
    t1, t2, t3, _ = build_split("test", [("US", 0.4), ("India", 0.4), ("France", 0.2)],
                                int(5000 * args.scale), 4242, 500000)
    write("test", t1, t2, t3)
    (OUT / "SYNTHETIC").write_text("generated by scripts/make_synthetic_data.py\n")
    print(f"train: {len(s1)} S1, {len(s2)} S2, {len(s3)} S3; test: {len(t1)} S1, {len(t2)} S2, {len(t3)} S3")


if __name__ == "__main__":
    main()
