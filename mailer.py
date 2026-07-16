import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


class Mailer:
    def __init__(self, outbox_path=None):
        self.smtp_host = os.environ.get("NBL_SMTP_HOST")
        self.smtp_port = int(os.environ.get("NBL_SMTP_PORT", "587"))
        self.smtp_user = os.environ.get("NBL_SMTP_USER")
        self.smtp_password = os.environ.get("NBL_SMTP_PASSWORD")
        self.smtp_from = os.environ.get("NBL_SMTP_FROM", "no-reply@neural-blocks.local")
        self.smtp_tls = os.environ.get("NBL_SMTP_TLS", "1") == "1"
        self.outbox_path = Path(outbox_path or ".data/mail-outbox.jsonl")

    @property
    def mode(self):
        return "smtp" if self.smtp_host else "file"

    def send(self, recipient, subject, text, metadata=None):
        message = {
            "to": recipient,
            "subject": subject,
            "text": text,
            "metadata": metadata or {},
        }
        if self.smtp_host:
            email = EmailMessage()
            email["From"] = self.smtp_from
            email["To"] = recipient
            email["Subject"] = subject
            email.set_content(text)
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as smtp:
                if self.smtp_tls:
                    smtp.starttls()
                if self.smtp_user:
                    smtp.login(self.smtp_user, self.smtp_password or "")
                smtp.send_message(email)
            return {"mode": "smtp"}

        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
        with self.outbox_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(message, ensure_ascii=False) + "\n")
        return {"mode": "file", "path": str(self.outbox_path)}


class MemoryMailer:
    def __init__(self):
        self.messages = []

    @property
    def mode(self):
        return "memory"

    def send(self, recipient, subject, text, metadata=None):
        self.messages.append({
            "to": recipient,
            "subject": subject,
            "text": text,
            "metadata": metadata or {},
        })
        return {"mode": "memory"}
