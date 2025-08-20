from flask import Flask, request, session
import random, string  

app = Flask(__name__)  
app.secret_key = "supersecretkey"  # Needed for session  

password = []  
for i in range(8):  
    password.append(random.choice(string.ascii_lowercase))  
final_password = ''.join(password)
 

@app.route("/", methods=["GET", "POST"])  
def login():
    if request.method == "GET":  
        session.clear()  
    error = ""  
    session.setdefault("attempts", 0)  
  
    if request.method == "POST":  
        if request.form.get("password") == final_password:  
            session["attempts"] = 0  # Reset on success  
            return "Welcome to the vault!"  
        else:  
            session["attempts"] += 1
            if session["attempts"] >= 3:  
                return "Too many attempts! Access denied." + "  Password was: " + final_password   
            error = f"Wrong password. Attempts left: {3 - session['attempts']}"  
  
    return f'''  
        {error}<br>  
        <form method="post">  
            Password: <input type="password" name="password">  
            <input type="submit" value="Enter">  
        </form>  
    '''  
  
if __name__ == "__main__":  
    app.run(debug=True)  