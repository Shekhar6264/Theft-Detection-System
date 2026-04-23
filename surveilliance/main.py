import os
import time

from flask import Flask, Response, jsonify, render_template, request

from camera import VideoCamera


def create_app():
    app = Flask(__name__)
    camera_instance = {"value": None}

    def get_camera():
        if camera_instance["value"] is None:
            camera_instance["value"] = VideoCamera()
        return camera_instance["value"]

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/health")
    def health():
        return jsonify({"ok": True})

    @app.route("/status")
    def status():
        return jsonify(get_camera().get_status())

    @app.route("/config/email", methods=["GET", "POST"])
    def email_config():
        camera = get_camera()

        if request.method == "GET":
            return jsonify(camera.get_email_config())

        payload = request.get_json(silent=True) or {}
        updated = camera.update_email_config(
            smtp_server=payload.get("smtp_server"),
            smtp_port=payload.get("smtp_port"),
            sender_email=payload.get("sender_email"),
            sender_password=payload.get("sender_password"),
            receiver_email=payload.get("receiver_email"),
        )
        return jsonify(updated)

    def gen():
        while True:
            frame = get_camera().get_frame()

            if not frame:
                time.sleep(0.1)
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

            time.sleep(0.03)

    @app.route("/video_feed")
    def video_feed():
        return Response(
            gen(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
    )
