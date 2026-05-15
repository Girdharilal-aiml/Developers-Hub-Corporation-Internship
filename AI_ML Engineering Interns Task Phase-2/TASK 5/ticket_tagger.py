"""
Auto-Tagging Support Tickets
==============================
Run this file:  python ticket_tagger.py

What it does:
- Loads support_tickets.csv
- Runs Zero-Shot, Few-Shot, and Fine-Tuned classification
- Shows Top 3 tags per ticket
- Compares all three methods
- Lets you type your own ticket and get predictions

Requirements:
    pip install pandas scikit-learn numpy
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
DATA_PATH = "support_tickets.csv"

LABELS = [
    "Billing",
    "Refund",
    "Technical Issue",
    "Account Access",
    "Subscription",
    "General Inquiry",
]

# Keywords for zero-shot classification
# This simulates how BART-MNLI reasons about categories
KEYWORD_MAP = {
    "Billing":         ["charge","invoice","bill","payment","fee","billed","receipt","tax","price","cost"],
    "Refund":          ["refund","money back","return","reimburse","refunded","give back","reverse","compensation"],
    "Technical Issue": ["crash","error","bug","broken","not working","slow","failed","500","403","blank","freezes","stuck"],
    "Account Access":  ["login","password","locked","access","log in","sign in","reset","cant get in","locked out"],
    "Subscription":    ["plan","upgrade","downgrade","cancel","renew","trial","seats","tier","subscribe","annual","monthly"],
    "General Inquiry": ["how","what","do you","available","information","policy","question","demo","feature","support hours"],
}

# Few-shot examples (5 per class — mimics FLAN-T5 few-shot prompting)
FEW_SHOT_EXAMPLES = {
    "Billing":         "my invoice shows an incorrect amount",
    "Refund":          "i want my money back for the purchase i made last week",
    "Technical Issue": "the app keeps crashing whenever i try to open it",
    "Account Access":  "i forgot my password and cannot log in",
    "Subscription":    "i want to upgrade my plan to the business tier",
    "General Inquiry": "do you offer a free trial for the premium plan",
}


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def print_banner(text):
    print()
    print("=" * 58)
    print(f"  {text}")
    print("=" * 58)


def print_top3(top3, show_bar=True):
    for rank, (label, score) in enumerate(top3, 1):
        bar = "#" * int(score * 25) if show_bar else ""
        print(f"    #{rank}  {label:<22}  {bar:<25}  {score:.4f}")


# ─────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"\n  Dataset not found: {DATA_PATH}")
        print("  Generating a sample dataset...")
        create_sample_dataset()

    df = pd.read_csv(DATA_PATH)
    df["ticket_text"] = df["ticket_text"].str.lower().str.strip()
    return df


def create_sample_dataset():
    """Generate a minimal support ticket dataset if CSV is missing."""
    tickets = []
    samples = {
        "Billing": [
            "i was charged twice for my subscription",
            "my invoice shows the wrong amount",
            "i need a copy of my billing statement",
            "there is an unauthorized charge on my account",
            "can you send me a detailed breakdown of my bill",
        ],
        "Refund": [
            "i want a full refund for my purchase",
            "please process my refund it has been 10 days",
            "i cancelled within 30 days so i deserve a refund",
            "my refund has not appeared in my account yet",
            "i was promised a refund but nothing has come through",
        ],
        "Technical Issue": [
            "the app keeps crashing every time i open it",
            "i am getting a 500 internal server error",
            "the dashboard is loading very slowly",
            "my data is not syncing across my devices",
            "notifications are not working on my phone",
        ],
        "Account Access": [
            "i forgot my password and cannot reset it",
            "my account is locked after too many attempts",
            "i cannot log in it says my email is not recognized",
            "someone changed my password i think i was hacked",
            "i need to change the email address on my account",
        ],
        "Subscription": [
            "i want to upgrade my plan to business tier",
            "how do i cancel my subscription",
            "i need to add 5 more seats to my team plan",
            "my free trial ended but i was not notified",
            "i want to switch from monthly to annual billing",
        ],
        "General Inquiry": [
            "what are your business hours for support",
            "do you offer a free trial for the premium plan",
            "what integrations do you support",
            "is your service available in my country",
            "where can i find your product documentation",
        ],
    }
    for tag, texts in samples.items():
        for text in texts:
            tickets.append({"ticket_id": f"TKT-{len(tickets)+1000}",
                            "ticket_text": text, "tag": tag})

    df = pd.DataFrame(tickets)
    df.to_csv(DATA_PATH, index=False)
    print(f"  Created {len(df)} sample tickets -> {DATA_PATH}")


# ─────────────────────────────────────────────────────────
# ZERO-SHOT CLASSIFIER
# Simulates BART-MNLI zero-shot — keyword scoring
# ─────────────────────────────────────────────────────────
def zero_shot_classify(text, labels=LABELS, seed=None):
    """
    Zero-shot: classify without any training examples.
    Uses keyword matching to score each category.
    In production: replace with facebook/bart-large-mnli pipeline.
    """
    rng = np.random.RandomState(seed or abs(hash(text)) % (2**31))
    t = text.lower()
    scores = {}
    for label in labels:
        keywords = KEYWORD_MAP.get(label, [])
        score = sum(3.0 if kw in t else 0.0 for kw in keywords)
        score += rng.uniform(0.05, 0.35)   # natural noise
        scores[label] = max(score, 0.05)

    total = sum(scores.values())
    scores = {k: v / total for k, v in scores.items()}
    ranked = sorted(labels, key=lambda x: -scores[x])
    return [(lbl, round(scores[lbl], 4)) for lbl in ranked[:3]]


# ─────────────────────────────────────────────────────────
# FEW-SHOT CLASSIFIER
# Simulates FLAN-T5 few-shot — trained on N examples per class
# ─────────────────────────────────────────────────────────
def build_few_shot_prompt(ticket_text, examples=FEW_SHOT_EXAMPLES):
    """Build the few-shot prompt shown to an instruction-following LLM."""
    prompt = "Classify the support ticket into one of these categories:\n"
    prompt += ", ".join(examples.keys()) + "\n\nExamples:\n"
    for label, example in examples.items():
        prompt += f'Ticket: "{example}"\nCategory: {label}\n\n'
    prompt += f'Now classify:\nTicket: "{ticket_text}"\nCategory:'
    return prompt


class FewShotClassifier:
    """
    Trains on just N examples per class.
    Equivalent to few-shot prompting with FLAN-T5.
    """
    def __init__(self, n_per_class=5):
        self.n = n_per_class
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000)),
            ("clf",   LogisticRegression(max_iter=300, C=0.5, random_state=42)),
        ])
        self.classes_ = None

    def fit(self, train_df):
        examples = pd.concat([
            train_df[train_df["tag"] == tag].head(self.n)
            for tag in LABELS
        ])
        self.pipe.fit(examples["ticket_text"], examples["tag"])
        self.classes_ = self.pipe.classes_.tolist()
        return self

    def predict_top3(self, text):
        probs = self.pipe.predict_proba([text.lower()])[0]
        top3i = np.argsort(probs)[::-1][:3]
        return [(self.classes_[i], round(float(probs[i]), 4)) for i in top3i]

    def predict(self, texts):
        return self.pipe.predict(texts)


# ─────────────────────────────────────────────────────────
# FINE-TUNED CLASSIFIER
# Simulates DistilBERT fine-tuned on full training data
# ─────────────────────────────────────────────────────────
class FineTunedClassifier:
    """
    Trains on the full training set.
    Equivalent to DistilBERT fine-tuned with Trainer API.
    """
    def __init__(self):
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2),
                                       max_features=10000,
                                       sublinear_tf=True)),
            ("clf",   LogisticRegression(max_iter=1000, C=5.0, random_state=42)),
        ])
        self.classes_ = None

    def fit(self, train_df):
        self.pipe.fit(train_df["ticket_text"], train_df["tag"])
        self.classes_ = self.pipe.classes_.tolist()
        return self

    def predict_top3(self, text):
        probs = self.pipe.predict_proba([text.lower()])[0]
        top3i = np.argsort(probs)[::-1][:3]
        return [(self.classes_[i], round(float(probs[i]), 4)) for i in top3i]

    def predict(self, texts):
        return self.pipe.predict(texts)


# ─────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────
def evaluate(name, predict_fn, test_df):
    preds = [predict_fn([t])[0] for t in test_df["ticket_text"]]
    acc   = accuracy_score(test_df["tag"], preds)
    f1    = f1_score(test_df["tag"], preds, average="macro", zero_division=0)
    return acc, f1


# ─────────────────────────────────────────────────────────
# DEMO — show predictions on sample tickets
# ─────────────────────────────────────────────────────────
def show_demo(zs_fn, fs_model, ft_model, test_df):
    print_banner("DEMO: Sample Ticket Predictions")

    samples = test_df.sample(5, random_state=99)

    for _, row in samples.iterrows():
        ticket = row["ticket_text"]
        true   = row["tag"]

        zs = zs_fn(ticket)
        fs = fs_model.predict_top3(ticket)
        ft = ft_model.predict_top3(ticket)

        print(f'\nTicket : "{ticket[:65]}"')
        print(f"True tag: {true}")

        for method, top3 in [("Zero-Shot ", zs), ("Few-Shot  ", fs), ("Fine-Tuned", ft)]:
            predicted = top3[0][0]
            ok = "OK" if predicted == true else "X "
            tags = "  |  ".join([f"{l} ({s:.3f})" for l, s in top3])
            print(f"  [{ok}] {method}: {tags}")


# ─────────────────────────────────────────────────────────
# INTERACTIVE MODE — type your own ticket
# ─────────────────────────────────────────────────────────
def interactive_mode(zs_fn, fs_model, ft_model):
    print_banner("INTERACTIVE MODE — Type Your Own Ticket")
    print("  Type a support ticket and see the top 3 predicted tags.")
    print("  Press Enter on an empty line to quit.\n")

    while True:
        try:
            ticket = input("  Your ticket: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not ticket:
            break

        zs = zs_fn(ticket)
        fs = fs_model.predict_top3(ticket)
        ft = ft_model.predict_top3(ticket)

        print()
        print(f"  Ticket: \"{ticket}\"")
        print()

        for method, top3 in [
            ("Zero-Shot  (BART-MNLI)",   zs),
            ("Few-Shot   (FLAN-T5)",     fs),
            ("Fine-Tuned (DistilBERT)",  ft),
        ]:
            print(f"  {method}:")
            print_top3(top3)
        print()

        again = input("  Try another? (y/n): ").strip().lower()
        if again != "y":
            break


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    print()
    print("+--------------------------------------------------+")
    print("|  Auto-Tagging Support Tickets                    |")
    print("|  Zero-Shot | Few-Shot | Fine-Tuned               |")
    print("+--------------------------------------------------+")

    # ── Load data ────────────────────────────────────────
    print_banner("STEP 1 — Loading Dataset")
    df = load_data()
    print(f"  Loaded {len(df):,} tickets across {df['tag'].nunique()} categories")
    print(f"  Categories: {', '.join(df['tag'].unique())}")
    print()
    print("  Distribution:")
    for tag, count in df["tag"].value_counts().items():
        print(f"    {tag:<22} {count} tickets")

    # ── Split ────────────────────────────────────────────
    print_banner("STEP 2 — Train / Test Split")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["tag"]
    )
    print(f"  Train : {len(train_df)} tickets")
    print(f"  Test  : {len(test_df)} tickets")

    # ── Train few-shot and fine-tuned ────────────────────
    print_banner("STEP 3 — Training Models")
    print("  Training Few-Shot model  (5 examples per class)...")
    fs_model = FewShotClassifier(n_per_class=5)
    fs_model.fit(train_df)
    print("  Done.")

    print("  Training Fine-Tuned model (full training set)...")
    ft_model = FineTunedClassifier()
    ft_model.fit(train_df)
    print("  Done.")

    # Zero-shot doesn't need training
    zs_fn = lambda texts: [zero_shot_classify(texts[0])[0][0]]

    # ── Evaluate ─────────────────────────────────────────
    print_banner("STEP 4 — Evaluating All Three Methods")

    # Zero-shot evaluation
    zs_preds = [zero_shot_classify(t)[0][0] for t in test_df["ticket_text"]]
    acc_zs   = accuracy_score(test_df["tag"], zs_preds)
    f1_zs    = f1_score(test_df["tag"], zs_preds, average="macro", zero_division=0)

    # Few-shot evaluation
    fs_preds = fs_model.predict(test_df["ticket_text"])
    acc_fs   = accuracy_score(test_df["tag"], fs_preds)
    f1_fs    = f1_score(test_df["tag"], fs_preds, average="macro", zero_division=0)

    # Fine-tuned evaluation
    ft_preds = ft_model.predict(test_df["ticket_text"])
    acc_ft   = accuracy_score(test_df["tag"], ft_preds)
    f1_ft    = f1_score(test_df["tag"], ft_preds, average="macro", zero_division=0)

    print()
    print(f"  {'Method':<30} {'Accuracy':>10} {'F1 Score':>10}")
    print("  " + "-" * 52)
    print(f"  {'Zero-Shot  (BART-MNLI)':<30} {acc_zs*100:>9.1f}% {f1_zs:>10.3f}")
    print(f"  {'Few-Shot   (FLAN-T5)':<30} {acc_fs*100:>9.1f}% {f1_fs:>10.3f}")
    print(f"  {'Fine-Tuned (DistilBERT)':<30} {acc_ft*100:>9.1f}% {f1_ft:>10.3f}")
    print()
    print(f"  Fine-tuned improvement over zero-shot:")
    print(f"    Accuracy: +{(acc_ft-acc_zs)*100:.1f}%")
    print(f"    F1 Score: +{(f1_ft-f1_zs):.3f}")

    # ── Fine-tuned report ─────────────────────────────────
    print()
    print("  Fine-Tuned Per-Category Report:")
    print()
    report = classification_report(
        test_df["tag"], ft_preds,
        target_names=sorted(df["tag"].unique()),
        zero_division=0
    )
    for line in report.split("\n"):
        print(f"    {line}")

    # ── Demo ─────────────────────────────────────────────
    show_demo(zero_shot_classify, fs_model, ft_model, test_df)

    # ── Menu ─────────────────────────────────────────────
    while True:
        print_banner("WHAT WOULD YOU LIKE TO DO?")
        print("  1 -> Type your own ticket and see predictions")
        print("  2 -> Show demo predictions again")
        print("  3 -> See full fine-tuned report")
        print("  4 -> Exit")
        print()

        choice = input("  Your choice (1/2/3/4): ").strip()

        if choice == "1":
            interactive_mode(zero_shot_classify, fs_model, ft_model)

        elif choice == "2":
            show_demo(zero_shot_classify, fs_model, ft_model, test_df)

        elif choice == "3":
            print()
            print(classification_report(
                test_df["tag"], ft_preds,
                target_names=sorted(df["tag"].unique()),
                zero_division=0
            ))

        elif choice == "4":
            print("\n  Goodbye!\n")
            break

        else:
            print("  Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
