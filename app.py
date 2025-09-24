import hashlib
import sqlite3
import os
from flask import Flask, request, session, render_template, render_template_string, url_for, redirect
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Configuration for file uploads
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'profile_pics')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def create_message_table():
    conn = sqlite3.connect('forms.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS messages (message TEXT, username TEXT, timestamp TEXT)")
    conn.commit()
    cursor.execute("PRAGMA table_info(messages)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'timestamp' not in columns:
        cursor.execute("ALTER TABLE messages ADD COLUMN timestamp TEXT")
        conn.commit()
        
    conn.close()


def create_users_table():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT, profile_pic TEXT)")
    conn.commit()

    cursor.execute("PRAGMA table_info(users)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'profile_pic' not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN profile_pic TEXT")
        conn.commit()
    
    conn.close()

create_message_table()
create_users_table()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/register", methods=["GET", "POST"])
def register():
    error = ""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        if result is None:
            cursor.execute("INSERT INTO users (username, password, profile_pic) VALUES (?, ?, ?)",
                           (username, password, "default.png"))
            conn.commit()
            conn.close()
            return render_template("success.html")
        else:
            error = "That username is already taken"
            conn.close()
    return render_template("register.html", error=error)


@app.route("/", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        conn.close()
        if user is None:
            error = "Invalid username or password."
        else:
            session["username"] = username
            return redirect(url_for('account'))
    return render_template("login.html", error=error)


@app.route("/account", methods=["GET", "POST"])
def account():
    if request.method == "POST":
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                
                conn = sqlite3.connect('users.db')
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET profile_pic = ? WHERE username = ?", (filename, session["username"]))
                conn.commit()
                conn.close()
                return redirect(url_for('account'))
    
    conn = sqlite3.connect('forms.db')
    cursor = conn.cursor()
    cursor.execute("SELECT username, message, timestamp FROM messages WHERE username = ? ORDER BY ROWID DESC", (session["username"],))
    messages = cursor.fetchall()
    conn.close()
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT profile_pic FROM users WHERE username = ?", (session["username"],))
    user_pic = cursor.fetchone()
    conn.close()
    
    profile_pic = user_pic[0] if user_pic and user_pic[0] else "default.png"
    
    return render_template("account.html", username=session["username"], messages=messages, profile_pic=profile_pic)


@app.route("/website", methods=["GET", "POST"])
def website():
    if request.method == "POST":
        message = request.form.get("message")
        if message:
            username = session.get("username")
            if username:
                conn = sqlite3.connect('forms.db')
                cursor = conn.cursor()
                # Insert the new message with the current timestamp
                cursor.execute("INSERT INTO messages (message, username, timestamp) VALUES (?, ?, ?)",
                               (message, username, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
                conn.close()
                return redirect(url_for('website'))

    conn = sqlite3.connect('forms.db')
    cursor = conn.cursor()
    cursor.execute("SELECT username, message, timestamp FROM messages ORDER BY ROWID DESC")
    messages = cursor.fetchall()
    conn.close()
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    profiles = {}
    for username, _, _ in messages:
        if username not in profiles:
            cursor.execute("SELECT profile_pic FROM users WHERE username = ?", (username,))
            user_pic = cursor.fetchone()
            profiles[username] = user_pic[0] if user_pic and user_pic[0] else "default.png"
    conn.close()
    
    return render_template("website.html", messages=messages, profiles=profiles, username=session.get("username"))


if __name__ == "__main__":
    app.run(debug=True)
