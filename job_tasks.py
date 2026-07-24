from pathlib import Path

from backend import NeuralBlocksBackend, iso_time, parse_json
from federation import (
    create_ags_line_item,
    fetch_nrps_members,
    post_ags_score,
)
from mailer import Mailer


class JobTaskRunner:
    def __init__(self, database_target, *, base_url, root=None):
        self.root = Path(root or Path(__file__).resolve().parent)
        self.mailer = Mailer(self.root / ".data" / "mail-outbox.jsonl")
        self.backend = NeuralBlocksBackend(
            database_target,
            mailer=self.mailer,
            base_url=base_url,
            expose_dev_tokens=False,
        )

    def __call__(self, job_type, payload):
        handlers = {
            "email.send": self.send_email,
            "lti.roster_sync": self.sync_lti_roster,
            "lti.grade_passback": self.send_lti_grade,
        }
        handler = handlers.get(job_type)
        if not handler:
            raise RuntimeError(f"Unsupported background job type: {job_type}")
        return handler(payload)

    def worker_auth(self, user_id):
        return self.backend.auth_for_user(
            user_id,
            {
                "ip": "background-worker",
                "userAgent": "Neural Blocks Job Worker",
            },
        )

    def send_email(self, payload):
        return self.mailer.send(
            payload["recipient"],
            payload["subject"],
            payload["text"],
            payload.get("metadata"),
        )

    def sync_lti_roster(self, payload):
        auth = self.worker_auth(payload["userId"])
        course_id = payload["courseId"]
        service = self.backend.get_lti_course_service(
            auth,
            course_id,
            include_private=True,
        )
        if not service["connected"] or not service["nrps"]["available"]:
            raise RuntimeError("NRPS membership service is unavailable")
        if not service["provider"]["enabled"]:
            raise RuntimeError("LTI provider is disabled")
        provider = service.pop("_provider")
        roster = fetch_nrps_members(
            provider,
            service["nrps"]["membershipsUrl"],
            service["nrps"]["scopes"],
        )
        result = self.backend.apply_lti_roster(
            auth,
            course_id,
            provider["id"],
            service["contextId"],
            roster["members"],
        )
        result["pages"] = roster["pages"]
        return result

    def send_lti_grade(self, payload):
        auth = self.worker_auth(payload["userId"])
        plan = self.backend.prepare_lti_grade_passback(
            auth,
            payload["submissionId"],
        )
        context = plan["context"]
        lineitem_url = plan["lineitemUrl"]
        if not lineitem_url:
            if not context.get("ags_lineitems_url"):
                raise RuntimeError("AGS line item service is unavailable")
            line_item = create_ags_line_item(
                plan["provider"],
                context["ags_lineitems_url"],
                parse_json(context.get("ags_scope_json"), []),
                plan["assignment"],
            )
            lineitem_url = self.backend.save_lti_line_item(
                auth,
                plan,
                line_item["url"],
            )
        else:
            self.backend.save_lti_line_item(auth, plan, lineitem_url)
        result = post_ags_score(
            plan["provider"],
            lineitem_url,
            parse_json(context.get("ags_scope_json"), []),
            {
                "userId": plan["studentSubject"],
                "scoreGiven": plan["score"],
                "scoreMaximum": 100,
                "timestamp": iso_time(),
                "comment": plan["feedback"],
            },
        )
        return self.backend.record_lti_grade_passback(
            auth,
            plan,
            lineitem_url,
            result,
        )
