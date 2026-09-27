from datetime import datetime
from functools import wraps
import json
import os
import pickle
import secrets
import sqlite3

from flask import (
    Flask,
    abort,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash


app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32),
    DATABASE=os.path.join(app.root_path, "fake_job_detector.db"),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

with open(os.path.join(app.root_path, "fake_job_model.pkl"), "rb") as file:
    model = pickle.load(file)

with open(os.path.join(app.root_path, "vectorizer.pkl"), "rb") as file:
    vectorizer = pickle.load(file)


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_error=None):
    database = g.pop("db", None)
    if database is not None:
        database.close()


def init_db():
    with sqlite3.connect(app.config["DATABASE"]) as database:
        database.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                job_description TEXT NOT NULL,
                prediction TEXT NOT NULL,
                risk_score INTEGER NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
                red_flags TEXT NOT NULL DEFAULT '[]',
                recommendation TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_predictions_user_created
                ON predictions(user_id, created_at DESC);
            """
        )


init_db()


@app.before_request
def protect_post_requests():
    if request.method == "POST":
        submitted_token = request.form.get("csrf_token", "")
        expected_token = session.get("csrf_token", "")
        if not expected_token or not secrets.compare_digest(submitted_token, expected_token):
            abort(400, description="Your session expired. Please refresh and try again.")


@app.context_processor
def inject_template_values():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return {"csrf_token": session["csrf_token"]}


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.user is None:
            flash("Please log in to continue.", "info")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


@app.before_request
def load_logged_in_user():
    user_id = session.get("user_id")
    g.user = None
    if user_id is not None:
        g.user = get_db().execute(
            "SELECT id, name, email, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if g.user is None:
            session.clear()


def detect_red_flags(text):
    text_lower = text.lower()
    red_flags = []
    keyword_groups = [
        (
            [
                "registration fee", "processing fee", "pay money", "pay fee",
                "payment required", "deposit", "joining fee", "security deposit",
            ],
            "Payment or registration fee requested",
        ),
        (
            [
                "earn ₹", "earn rs", "earn 50000", "earn 100000", "huge salary",
                "huge money", "guaranteed income", "easy money",
            ],
            "Unrealistic or guaranteed income claim",
        ),
        (
            [
                "apply now", "limited vacancies", "urgent hiring", "join immediately",
                "immediate joining", "act now", "hurry",
            ],
            "Urgent or pressure-based language",
        ),
        (
            ["whatsapp", "telegram", "contact me on whatsapp", "message on telegram"],
            "Unusual communication method mentioned",
        ),
        (
            ["bank account", "bank details", "credit card", "debit card", "otp", "aadhaar", "pan card"],
            "Sensitive personal or financial information requested",
        ),
    ]

    for keywords, label in keyword_groups:
        if any(keyword in text_lower for keyword in keywords):
            red_flags.append(label)
    if "no experience required" in text_lower:
        red_flags.append("No experience required for potentially high earnings")
    return red_flags


def calculate_risk(model, input_vector, red_flags):
    probabilities = model.predict_proba(input_vector)[0]
    fake_class_index = list(model.classes_).index(1)
    model_score = float(probabilities[fake_class_index]) * 100
    flag_signal = min(len(red_flags) * 25, 100)
    return round(max(0, min(100, model_score * 0.85 + flag_signal * 0.15)))


def analyze_job(job_description):
    input_vector = vectorizer.transform([job_description])
    result = int(model.predict(input_vector)[0])
    red_flags = detect_red_flags(job_description)
    risk_score = calculate_risk(model, input_vector, red_flags)
    if result == 1:
        prediction = "Potentially Fake Job"
        recommendation = (
            "Be careful with this job posting. Do not pay money or share sensitive "
            "personal information until the employer is independently verified."
        )
    else:
        prediction = "Likely Genuine Job"
        recommendation = (
            "This posting appears relatively low-risk. Verify the employer and "
            "job details through official channels before applying."
        )
    return {
        "prediction": prediction,
        "risk_score": risk_score,
        "red_flags": red_flags,
        "recommendation": recommendation,
    }


@app.route("/")
def home():
    if g.user:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if g.user:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirmation = request.form.get("confirm_password", "")
        if not name or not email or not password or not confirmation:
            flash("Please complete every field.", "error")
        elif "@" not in email or "." not in email.rsplit("@", 1)[-1]:
            flash("Enter a valid email address.", "error")
        elif len(password) < 8:
            flash("Use a password with at least 8 characters.", "error")
        elif password != confirmation:
            flash("The passwords do not match.", "error")
        else:
            database = get_db()
            existing_user = database.execute(
                "SELECT id FROM users WHERE email = ?", (email,)
            ).fetchone()
            if existing_user:
                flash("An account with that email already exists.", "error")
            else:
                database.execute(
                    "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                    (name, email, generate_password_hash(password)),
                )
                database.commit()
                flash("Your account is ready. Please log in.", "success")
                return redirect(url_for("login"))
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = get_db().execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
        if user is None or not check_password_hash(user["password"], password):
            flash("Email or password was not recognized.", "error")
        else:
            session.clear()
            session["user_id"] = user["id"]
            session["csrf_token"] = secrets.token_urlsafe(32)
            flash("Welcome back.", "success")
            return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


@app.route("/dashboard")
@login_required
def dashboard():
    database = get_db()
    stats = database.execute(
        """
        SELECT COUNT(*) AS total,
               COALESCE(SUM(prediction = 'Potentially Fake Job'), 0) AS fake,
               COALESCE(SUM(prediction = 'Likely Genuine Job'), 0) AS genuine,
               COALESCE(ROUND(AVG(risk_score)), 0) AS average_risk
        FROM predictions WHERE user_id = ?
        """,
        (g.user["id"],),
    ).fetchone()
    recent = database.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC, id DESC LIMIT 5",
        (g.user["id"],),
    ).fetchall()
    return render_template("dashboard.html", stats=stats, recent=recent)


@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    result = None
    user_input = ""
    if request.method == "POST":
        user_input = request.form.get("job_description", "").strip()
        if not user_input:
            flash("Paste a job description to analyze.", "error")
        else:
            result = analyze_job(user_input)
            get_db().execute(
                """
                INSERT INTO predictions
                    (user_id, job_description, prediction, risk_score, red_flags, recommendation)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    g.user["id"], user_input, result["prediction"], result["risk_score"],
                    json.dumps(result["red_flags"]), result["recommendation"],
                ),
            )
            get_db().commit()
            flash("Analysis saved to your history.", "success")
    return render_template("analyze.html", result=result, user_input=user_input)


@app.route("/history")
@login_required
def history():
    rows = get_db().execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC, id DESC",
        (g.user["id"],),
    ).fetchall()
    return render_template("history.html", predictions=rows)


@app.route("/profile")
@login_required
def profile():
    total = get_db().execute(
        "SELECT COUNT(*) FROM predictions WHERE user_id = ?", (g.user["id"],)
    ).fetchone()[0]
    return render_template(
        "profile.html", total=total, joined=datetime.fromisoformat(g.user["created_at"])
    )


if __name__ == "__main__":
    app.run(debug=True)