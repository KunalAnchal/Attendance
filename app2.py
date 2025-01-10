from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy, session
from sqlalchemy import UniqueConstraint
import os
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
import face_recognition
import psycopg2
import pyttsx3
from datetime import datetime
import threading, queue
import numpy as np
import threading
from gtts import gTTS
import pygame
import logging
import base64
import io
import json
import yaml
from influxdb_client import InfluxDBClient, Point, WriteOptions
import mysql.connector
import requests
from requests.exceptions import RequestException




app = Flask(__name__)
app.config['SECRET_KEY'] = 'yghuihYGKH98HK9Y87877IBy8Hy9yI8Ggihojdjk@@@++==huy8gdhce9fyievg38febvjw8udy33bciencwfy83a'

# Configure the database URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:Admin%408084@localhost/Employee'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
# login_manager = LoginManager(app)
# login_manager.login_view = "login"
speech_queue = queue.Queue()
engine = pyttsx3.init()



lock = threading.Lock()
audio_lock = threading.Lock()



# # Initialize logging
# logging.basicConfig(
#     filename="face_greeting.log",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
#
# logging.info("Application started.")


def get_db_connection():
    conn = psycopg2.connect(
        host="localhost",        # Assuming the database is on the same machine
        database="Employee",  # Your database name
        user="postgres",         # Your username
        password="Admin@8084"     # Your password
    )
    return conn


def get_greeting(time_of_day, language):
    greetings = {
        "english": {"morning": "Good morning", "afternoon": "Good afternoon", "evening": "Good evening"},
        "arabic": {"morning": "صباح الخير", "afternoon": "مساء الخير", "evening": "مساء الخير"}
    }
    return greetings.get(language, greetings["english"]).get(time_of_day, "Hello")


class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_name = db.Column(db.String(100))
    employee_id = db.Column(db.String(50), nullable=False)
    employee_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    department = db.Column(db.String(100))
    date_of_birth = db.Column(db.Date, nullable=False)
    marital_status = db.Column(db.String(50))
    marriage_anniversary = db.Column(db.Date)
    country = db.Column(db.String(100))
    preferred_language = db.Column(db.String(50))
    designation = db.Column(db.String(100))
    shift = db.Column(db.String(50))
    manager_name = db.Column(db.String(100))
    gender = db.Column(db.String(10), nullable=False)
    joining_date = db.Column(db.Date, nullable=False)
    face_encoding = db.Column(db.LargeBinary, nullable=False)



class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_name = db.Column(db.String(100), nullable=False)
    employee_id = db.Column(db.String(50), nullable=False)
    detected_time = db.Column(db.DateTime, nullable=False)


def load_encodings():
    conn = get_db_connection()  # Step 1: Establish a database connection
    cursor = conn.cursor()  # Step 2: Create a cursor object to interact with the database
    cursor.execute("SELECT employee_id, employee_name, face_encoding FROM employee")  # Step 3: Execute the query to fetch employee ID, name, and face encoding
    rows = cursor.fetchall()  # Step 4: Fetch all rows from the result set
    encodings = {}  # Step 5: Initialize an empty dictionary to store the encodings
    for row in rows:  # Step 6: Loop through each row
        encoding = np.frombuffer(row[2], dtype=np.float64)  # Decode the binary face encoding
        encodings[row[1]] = encoding  # Map employee name to the corresponding face encoding
        encodings[row[0]] = encoding  # Map employee_id to the corresponding face encoding
    conn.close()  # Step 7: Close the database connection
    return encodings



def speak_greeting():
    while True:
        message_to_speak = speech_queue.get()  # Wait for a new message
        if message_to_speak is None:
            break  # Exit when a None value is put in the queue
        engine.say(message_to_speak)
        engine.runAndWait()

# Start a single background thread to handle speech
speech_thread = threading.Thread(target=speak_greeting)
speech_thread.daemon = True
speech_thread.start()


# def speak_greeting_async(message_to_speak):
#     speech_queue.put(message_to_speak)  # Add the message to the queue

def speak_greeting_async(message_to_speak):
    with audio_lock:  # Ensure only one thread accesses this block
        speech_queue.put(message_to_speak)


def play_audio_with_pygame(text):
    try:
        with audio_lock:  # Ensure only one audio plays at a time
            audio_file = "arabic_text.mp3"
            tts = gTTS(text=text, lang='ar', slow=False)
            tts.save(audio_file)

            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                continue
            logging.info("Audio played successfully.")
    except Exception as e:
        logging.error(f"Error playing audio with pygame: {e}")
    finally:
        pygame.mixer.quit()
        if os.path.exists(audio_file):
            try:
                os.remove(audio_file)
                logging.info("Temporary audio file deleted.")
            except Exception as e:
                logging.warning(f"Failed to delete temporary audio file {audio_file}: {e}")


def get_absent_employees():
    """
    Fetches the list of employees who have not marked their attendance for today,
    including employee_name, employee_id, and email.
    """
    today = datetime.now().date()  # Get today's date

    with app.app_context():  # Ensure the app context is active
        # Query all employees
        all_employees = Employee.query.all()

        absent_employees = []
        for employee in all_employees:
            # Check if the employee's attendance for today exists
            attendance_record = Attendance.query.filter(
                Attendance.employee_id == employee.employee_id,
                db.func.date(Attendance.detected_time) == today
            ).first()

            if attendance_record is None:
                # Include email in the absent employee data
                absent_employees.append({
                    "employee_id": employee.employee_id,
                    "employee_name": employee.employee_name,
                    "email": employee.email
                })

    return absent_employees


domain = 'sandbox6f9214b7e8fb49ac91b0415c2681883c.mailgun.org'
api_key = '045a976151cf6cb983c82354dda93af6-78f6ccbe-649b3169'


def send_email(subject, body, recipient_email):
    mailgun_url = f"https://api.mailgun.net/v3/{domain}/messages"
    mailgun_api_key = ("api", api_key)

    # Ensure recipient_email is valid
    if not recipient_email:
        print("Error: Invalid recipient email")
        return

    email_data = {
        "from": 'kharsh854311@gmail.com',  # You can set this as an environment variable too if needed
        "to": recipient_email,
        "subject": subject,
        "text": body
    }

    try:
        # Send the email request
        response = requests.post(mailgun_url, auth=mailgun_api_key, data=email_data,
                                 verify=True)  # Enabling SSL verification

        # Check the response from the Mailgun API
        if response.status_code == 200:
            print("Email sent successfully!")
        else:
            print(f"Failed to send email. Status code: {response.status_code}")
            print(f"Response text: {response.text}")

    except RequestException as e:
        # Catching any request-related issues
        print(f"An error occurred while sending the email: {e}")
    except Exception as e:
        # Catch any other exceptions
        print(f"An unexpected error occurred: {e}")




def gen(custom_message, language):
    cap = cv2.VideoCapture(0)
    encodings = load_encodings()  # Load pre-stored encodings as a dictionary {name: encoding}
    played_messages = {}  # Track last greeting time for each person

    current_time = datetime.now()

    # Define the specific time (12:00:00)
    target_time = datetime.strptime('12:00:00', '%H:%M:%S').time()

    if current_time.time() == target_time:
        print("It's 12:00 PM or later. Checking for absent employees...")

        # Get absent employees
        absent_employees = get_absent_employees()
        print(absent_employees)
        if absent_employees:
            for employee in absent_employees:
                subject = "Attendance Reminder"
                body = f"Dear {employee['employee_name']},\n\nYou have not marked your attendance for today. Please ensure that you log your attendance.\n\nBest regards,\nYour Company"
                send_email(subject, body, employee['email'])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Calculate distances between the detected face and all known encodings
            distances = face_recognition.face_distance(list(encodings.values()), face_encoding)
            best_match_index = np.argmin(distances)  # Index of the smallest distance
            name = "Unknown"
            employee_id = None

            # Match the detected face with a confidence threshold
            if distances[best_match_index] < 0.5:  # Confidence threshold
                name = list(encodings.keys())[best_match_index]  # Retrieve the corresponding name
                employee_id = get_employee_id(name)  # Assume a function that fetches employee ID by name

            print(f"Recognized Name: {name}, Employee ID: {employee_id}")

            # Log attendance only if a known face is recognized
            if name != "Unknown" and employee_id:
                log_attendance_to_db(employee_id, name)
                print("Attendance data saved.")
            else:
                print("Attendance is not uploaded.")

            # Determine the time of day for personalized greeting
            current_time = datetime.now().hour
            if 5 <= current_time < 12:
                time_of_day = 'morning'
            elif 12 <= current_time < 17:
                time_of_day = 'afternoon'
            else:
                time_of_day = 'evening'

            greeting = f"{get_greeting(time_of_day, language)}, {name}!" if name != "Unknown" else f"Hello, {time_of_day}!"
            message_to_speak = f"{greeting} {custom_message}"





            # Play audio message only if it hasn't been played recently
            current_timestamp = datetime.now().timestamp()
            last_played = played_messages.get(name, 0)

            if name != "Unknown" and (current_timestamp - last_played > 15):
                if language == "arabic":
                    play_audio_with_pygame(message_to_speak)
                else:
                    speak_greeting_async(message_to_speak)
                played_messages[name] = current_timestamp  # Update the timestamp

            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')



    cap.release()


@app.route("/submit", methods=["POST"])
def submit():
    try:
        logging.debug('Form submission started.')

        # Extract form data
        company_name = request.form.get('company_name')
        employee_id = request.form.get('employee_id')
        employee_name = request.form.get('employee_name')
        email = request.form.get('email')
        department = request.form.get('department')
        date_of_birth = request.form.get('date_of_birth')
        marital_status = request.form.get('marital_status')
        marriage_anniversary = request.form.get('marriage_anniversary')
        country = request.form.get('country')
        preferred_language = request.form.get('preferred_language')
        designation = request.form.get('designation')
        shift = request.form.get('shift')
        manager_name = request.form.get('manager_name')
        gender = request.form.get('gender')
        joining_date = request.form.get('joining_date')
        logging.debug(f'Extracted form data: {locals()}')

        # Validate required fields
        if not (employee_id and employee_name and date_of_birth and gender and joining_date):
            logging.warning('Missing required fields.')
            flash('Please fill in all required fields.', 'danger')
            return render_template('administration.html')

        # Handle image input
        face_encoding_bytes = None
        if 'face_snapshot' in request.form and request.form.get('face_snapshot'):
            # Process captured image from camera
            image_data = request.form.get('face_snapshot').split(',')[1]  # Remove data URL prefix
            image_bytes = base64.b64decode(image_data)
            img = face_recognition.load_image_file(io.BytesIO(image_bytes))
        elif 'upload_snapshot' in request.files and request.files['upload_snapshot'].filename:
            # Process uploaded image file
            file = request.files['upload_snapshot']
            img = face_recognition.load_image_file(file)
        else:
            logging.warning('No image provided.')
            flash('No image provided. Please capture or upload an image.', 'danger')
            return jsonify({"status": "error", "message": f"No image provided:"}), 500

        logging.debug(f'Image loaded: {img}')

        # Extract face encodings
        face_encodings = face_recognition.face_encodings(img)
        if not face_encodings:
            logging.warning('No face detected in the provided image.')
            flash('No face detected in the provided image. Please try again.', 'danger')
            return jsonify({"status": "error", "message": f"No face detected in the provided image:"}), 500

        face_encoding_bytes = face_encodings[0].tobytes()
        logging.debug(f'Face encoding extracted: {face_encoding_bytes[:10]}...')  # Log the first few bytes

        # Save to database
        employee = Employee(
            company_name=company_name,
            employee_id=employee_id,
            employee_name=employee_name,
            email=email,
            department=department,
            date_of_birth=datetime.strptime(date_of_birth, '%Y-%m-%d'),
            marital_status=marital_status,
            marriage_anniversary=datetime.strptime(marriage_anniversary, '%Y-%m-%d') if marriage_anniversary else None,
            country=country,
            preferred_language=preferred_language,
            designation=designation,
            shift=shift,
            manager_name=manager_name,
            gender=gender,
            joining_date=datetime.strptime(joining_date, '%Y-%m-%d'),
            face_encoding=face_encoding_bytes
        )

        db.session.add(employee)
        db.session.commit()
        logging.info('Employee data saved successfully.')
        flash('Employee data successfully saved!', 'success')
        return jsonify({"status": "success", "message": "Employee data saved successfully!"}), 200

    except Exception as e:
        logging.error(f'Error during form submission: {e}')
        flash(f'An error occurred: {e}', 'danger')
        return jsonify({"status": "error", "message": f"Error saving data: {str(e)}"}), 500



# Route to stream the video
def generate_video_stream():
    camera = cv2.VideoCapture(0)  # Use the first webcam
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Yield frame in byte format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@app.route('/video_feed1')
def video_feed1():
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')






def get_employee_id(name):
    # Database connection settings
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Query to fetch the employee ID based on the name
        cursor.execute("SELECT employee_id FROM employee WHERE employee_name = %s", (name,))
        result = cursor.fetchone()
        return result[0] if result else None  # Return the employee_id if found, else None
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


def log_attendance_to_db(employee_id, employee_name):
    """Logs attendance data to the Attendance table in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    current_time = datetime.now()

    # Insert the attendance record
    try:
        cursor.execute(
            "INSERT INTO attendance (employee_id, employee_name, detected_time) VALUES (%s, %s, %s)",
            (employee_id, employee_name, current_time)
        )
        conn.commit()
    except Exception as e:
        print(f"Error logging attendance: {e}")
    finally:
        conn.close()



from datetime import datetime


def calculate_working_hours(employee_id, start_date, end_date):
    # Establish a database connection
    conn = get_db_connection()
    cursor = conn.cursor()

    # # Query to fetch attendance data
    # query = """
    #     SELECT employee_name, employee_id, DATE(detected_time) AS date,
    #            MIN(detected_time) AS first_detect,
    #            MAX(detected_time) AS last_detect
    #     FROM attendance
    #     WHERE employee_id = %s AND DATE(detected_time) BETWEEN %s AND %s
    #     GROUP BY employee_name, employee_id, DATE(detected_time)
    #     ORDER BY employee_name, employee_id;
    # """

    # Execute the query
    cursor.execute(
        "SELECT employee_name, employee_id, DATE(detected_time) AS date, MIN(detected_time) AS first_detect,MAX(detected_time) AS last_detect FROM attendance WHERE employee_id = %s AND DATE(detected_time) BETWEEN %s AND %s GROUP BY employee_name, employee_id, DATE(detected_time) ORDER BY employee_name, employee_id;",
        (employee_id, start_date, end_date))

    # Fetch the data
    data = cursor.fetchall()
    print("Fetched data:", data)  # Print fetched data to verify

    # Process the fetched data
    report = []
    for row in data:
        employee_name, employee_id, date, first_detect, last_detect = row

        # Ensure no conversion, as first_detect and last_detect are already datetime objects
        duration_seconds = (last_detect - first_detect).total_seconds()
        working_hours = duration_seconds / 3600  # Convert to hours

        # Append the calculated report
        report.append({
            "employee_name": employee_name,
            "employee_id": employee_id,
            "date": date,
            "working_hours": round(working_hours, 2)  # Rounded to 2 decimal places
        })

    # Close the database connection
    conn.close()
    return report


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/administration')
def administration():
    return render_template('administration.html')


# Route to display the form and report
@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    report = None
    if request.method == "POST":
        employee_name = request.form.get("employeeName")
        employee_id = request.form.get("employeeId")
        start_date = request.form.get("startDate")
        end_date = request.form.get("endDate")

        print(f"Received input: {employee_name}, {employee_id}, {start_date}, {end_date}")
        print(start_date)
        # employee_id = int(employee_id)
        print(type(employee_name))
        print(type(employee_id))
        print(type(start_date))
        print(type(end_date))


        if employee_name and employee_id and start_date and end_date:
            report = calculate_working_hours(employee_id, start_date, end_date)
            print("Report:", report)

            if not report or len(report) == 0:
                flash('No records found for the specified criteria.', 'danger')

            return jsonify(report)  # Return JSON response

    return render_template("attendance.html", report=report)





# # Route to handle form submission
# @app.route('/')
# def form():
#     return render_template('form.html')

















@app.route('/facial_recognition', methods=['GET', 'POST'])
def facial_recognition():
    return render_template('facial_recognition.html')

    # elif request.method == 'POST':
    #     custom_message = request.form['custom_message']
    #     language = request.form['language']
    #
    #     return Response(gen(custom_message, language), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    custom_message = request.args.get('custom_message', '')
    language = request.args.get('language', 'english')
    return Response(gen(custom_message, language), mimetype='multipart/x-mixed-replace; boundary=frame')






class Camera(db.Model):
    __tablename__ = 'cameras'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    ip = db.Column(db.String(100), nullable=False)
    port = db.Column(db.Integer, nullable=False)
    camera_number = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

    __table_args__ = (
        UniqueConstraint('user_id', 'camera_number', name='unique_user_camera_number'),
    )


class Model(db.Model):
    __tablename__ = 'models'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    file = db.Column(db.LargeBinary, nullable=False)


class CameraModel(db.Model):
    __tablename__ = 'camera_models'

    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, db.ForeignKey('cameras.id'), nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey('models.id'), nullable=False)

    __table_args__ = (
        UniqueConstraint('camera_id', 'model_id', name='unique_camera_model'),
    )


@app.route('/form')
def form():
    """Render the Configure Camera page."""
    return render_template('form.html')

@app.route('/configure_camera')
def configure_camera():
    """Render the Configure Camera page."""
    return render_template('configure_camera.html')


@app.route('/upload_model')
def upload_model_page():
    """Render the Upload Model page."""
    return render_template('upload_model.html')


@app.route('/assign_models')
def assign_models():

    # If user is admin, show the page.
    cameras = Camera.query.all()
    models = Model.query.all()
    return render_template('assign_models.html', cameras=cameras, models=models)


# Endpoint to add a camera
@app.route('/add_camera', methods=['POST'])
def add_camera():
    data = request.get_json()
    user_id = data.get('user_id')
    camera_number = data.get('camera_number')

    existing_camera = Camera.query.filter_by(user_id=user_id, camera_number=camera_number).first()
    if existing_camera:
        return jsonify({"message": "Camera already added."}), 400

    new_camera = Camera(
        name=data.get('name'),
        ip=data.get('ip'),
        port=data.get('port'),
        camera_number=camera_number,
        user_id=user_id,
        password=data.get('password')
    )

    db.session.add(new_camera)
    db.session.commit()

    return jsonify({"message": "Camera added successfully!", "camera_id": new_camera.id}), 201


# Endpoint to list all cameras
@app.route('/list_cameras')
def list_cameras():
    cameras = Camera.query.all()
    return jsonify([{
        'id': camera.id,
        'name': camera.name,
        'ip': camera.ip,
        'port': camera.port,
        'camera_number': camera.camera_number,
        'user_id': camera.user_id,
        'password': camera.password
    } for camera in cameras])


# Endpoint to upload a YOLO model
@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'model_file' not in request.files:
        return jsonify({"message": "No file part in the request"}), 400

    file = request.files['model_file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    # Ensure file does not already exist
    existing_model = Model.query.filter_by(name=file.filename).first()
    if existing_model:
        return jsonify({"message": f"Model {file.filename} already exists."}), 409

    # Save the model to the database
    new_model = Model(name=file.filename, file=file.read())
    db.session.add(new_model)
    db.session.commit()

    return jsonify({"message": "Model uploaded successfully!", "model_id": new_model.id}), 201


@app.route('/list_models')
def list_models():
    models = Model.query.all()
    return jsonify([{'id': model.id, 'name': model.name} for model in models])


@app.route('/assign_models', methods=['POST'])
def assign_models_post():
    data = request.get_json()
    camera_id = data.get('camera_id')
    model_ids = data.get('model_ids', [])

    if not camera_id:
        return jsonify({"message": "camera_id is required"}), 400

    # Remove all existing mappings for this camera
    CameraModel.query.filter_by(camera_id=camera_id).delete()

    # Add new mappings
    for model_id in model_ids:
        new_mapping = CameraModel(camera_id=camera_id, model_id=model_id)
        db.session.add(new_mapping)

    db.session.commit()
    return jsonify({"message": "Models assigned successfully!"}), 201



@app.route('/get_assigned_models/<int:camera_id>')
def get_assigned_models(camera_id):
    assigned_models = CameraModel.query.filter_by(camera_id=camera_id).all()
    return jsonify([
        {'id': cm.model_id, 'name': db.session.get(Model, cm.model_id).name}
        for cm in assigned_models
    ])


# Helper function to save and retrieve model file
def get_model_file(model_id):
    model = db.session.get(Model, model_id)
    if not model:
        raise ValueError("Model not found")

    # Save the model to a temporary location if not already saved
    model_path = f"models/{model.name}"
    if not os.path.exists("models"):
        os.makedirs("models")

    if not os.path.exists(model_path):
        with open(model_path, 'wb') as f:
            f.write(model.file)
    return model_path


# Endpoint to stream video with assigned models
@app.route('/stream/<int:camera_id>')
def stream(camera_id):
    camera = db.session.get(Camera, camera_id)
    if not camera:
        return "Camera not found", 404

    # Step 3: Retrieve the models currently assigned to this camera
    assigned_models = CameraModel.query.filter_by(camera_id=camera_id).all()
    model_paths = [get_model_file(cm.model_id) for cm in assigned_models]

    # Construct the RTSP URL
    rtsp_url = f"rtsp://{camera.user_id}:{camera.password}@{camera.ip}:{camera.port}/{camera.camera_number}"

    def generate_frames():
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            yield b"Failed to open RTSP stream"
            return

        # Load YOLOv8 models dynamically
        from ultralytics import YOLO
        models = [YOLO(model_path) for model_path in model_paths]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Step 4: Process each model on the frame
            for model in models:
                results = model(frame)
                for result in results:
                    annotated_frame = result.plot()  # Draw detections on the frame
                    frame = annotated_frame

            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

with app.app_context():
    db.create_all()  # Create database tables



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8090,debug=False)
