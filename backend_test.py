#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

from backend import ApiError, NeuralBlocksBackend
from mailer import MemoryMailer


class BackendTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.mailer = MemoryMailer()
        self.backend = NeuralBlocksBackend(
            Path(self.temp.name) / "test.db",
            mailer=self.mailer,
        )

    def tearDown(self):
        self.temp.cleanup()

    def register_institution(self, suffix):
        registration = self.backend.register({
            "email": f"professor-{suffix}@example.edu",
            "password": "correct-horse-battery-staple",
            "displayName": f"Professor {suffix}",
            "createInstitution": True,
            "institutionName": f"University {suffix}",
            "institutionSlug": f"university-{suffix}",
        })
        self.assertFalse(registration["user"]["emailVerified"])
        self.backend.verify_email({"token": registration["devVerificationToken"]})
        return registration

    def test_auth_tenant_isolation_and_classroom_flow(self):
        tenant_a = self.register_institution("alpha")
        tenant_b = self.register_institution("beta")
        auth_a = self.backend.authenticate(tenant_a["sessionToken"])
        auth_b = self.backend.authenticate(tenant_b["sessionToken"])
        self.assertNotEqual(auth_a["tenant"]["id"], auth_b["tenant"]["id"])

        login_a = self.backend.login({
            "email": "professor-alpha@example.edu",
            "password": "correct-horse-battery-staple",
        })
        self.assertEqual(login_a["user"]["tenantId"], auth_a["tenant"]["id"])

        course_a = self.backend.create_course(auth_a, {
            "name": "AI Foundations",
            "code": "AI101",
            "term": "2026-2",
        })
        course_b = self.backend.create_course(auth_b, {
            "name": "Machine Learning",
            "code": "ML201",
            "term": "2026-2",
        })
        self.assertEqual(len(self.backend.list_courses(auth_a)), 1)
        self.assertEqual(len(self.backend.list_courses(auth_b)), 1)
        self.assertNotEqual(course_a["id"], course_b["id"])

        student_registration = self.backend.register({
            "email": "student-alpha@example.edu",
            "password": "student-password-1234",
            "displayName": "Student Alpha",
            "createInstitution": False,
            "institutionJoinCode": auth_a["tenant"]["joinCode"],
        })
        self.backend.verify_email({
            "token": student_registration["devVerificationToken"],
        })
        student_auth = self.backend.authenticate(student_registration["sessionToken"])
        joined_course = self.backend.join_course(
            student_auth,
            {"joinCode": course_a["joinCode"]},
        )
        self.assertIsNone(joined_course["joinCode"])
        self.assertIsNone(self.backend.list_courses(student_auth)[0]["joinCode"])
        self.assertEqual(len(self.backend.list_courses(student_auth)), 1)

        with self.assertRaises(ApiError) as tenant_error:
            self.backend.list_assignments(student_auth, course_b["id"])
        self.assertEqual(tenant_error.exception.status, 404)

        assignment = self.backend.create_assignment(auth_a, course_a["id"], {
            "title": "XOR Lab",
            "instructions": "Reach validation accuracy 80%.",
            "requiredFamily": "mlp",
            "targetAccuracy": 0.8,
            "starterSnapshot": {
                "model": {"family": "mlp"},
                "result": {"validationAccuracy": 0.5},
            },
        })
        self.assertEqual(len(self.backend.list_assignments(student_auth, course_a["id"])), 1)

        snapshot = {
            "model": {"family": "mlp", "parameterCount": 33},
            "result": {"validationAccuracy": 0.91, "validationLoss": 0.2},
        }
        project = self.backend.save_project(student_auth, course_a["id"], {
            "name": "Student XOR",
            "snapshot": snapshot,
        })
        self.assertEqual(project["versionCount"], 1)
        loaded_project = self.backend.get_project(student_auth, project["id"])
        self.assertEqual(
            loaded_project["latestSnapshot"]["result"]["validationAccuracy"],
            0.91,
        )

        submission = self.backend.submit_assignment(student_auth, assignment["id"], {
            "projectId": project["id"],
            "snapshot": snapshot,
        })
        self.assertTrue(submission["autoEvaluation"]["passed"])
        self.assertEqual(submission["attempt"], 1)

        graded = self.backend.grade_submission(auth_a, submission["id"], {
            "score": 94,
            "feedback": "Reproducible experiment.",
        })
        self.assertEqual(graded["status"], "graded")
        self.assertEqual(graded["score"], 94)
        self.assertEqual(
            self.backend.list_submissions(student_auth, course_a["id"])[0]["score"],
            94,
        )
        self.assertEqual(len(self.backend.list_submissions(auth_b, course_b["id"])), 0)

    def test_invitations_roster_audit_password_reset_and_provider_state(self):
        registration = self.register_institution("operations")
        admin = self.backend.authenticate(registration["sessionToken"])
        course = self.backend.create_course(admin, {
            "name": "Production AI",
            "code": "PAI301",
            "term": "2026-2",
        })

        professor_invitation = self.backend.create_invitation(admin, {
            "email": "invited-professor@example.edu",
            "role": "professor",
            "courseId": course["id"],
        })
        professor_registration = self.backend.accept_invitation({
            "token": professor_invitation["devInvitationToken"],
            "displayName": "Invited Professor",
            "password": "invited-professor-password",
        })
        self.assertEqual(professor_registration["user"]["role"], "professor")
        self.assertTrue(professor_registration["user"]["emailVerified"])

        student_invitation = self.backend.create_invitation(admin, {
            "email": "invited-student@example.edu",
            "role": "student",
            "courseId": course["id"],
        })
        student_registration = self.backend.accept_invitation({
            "token": student_invitation["devInvitationToken"],
            "displayName": "Invited Student",
            "password": "invited-student-password",
        })
        members = self.backend.list_course_members(admin, course["id"])
        self.assertEqual(len(members), 3)
        self.assertEqual(
            {member["courseRole"] for member in members},
            {"instructor", "student"},
        )

        self.backend.remove_course_member(
            admin,
            course["id"],
            student_registration["user"]["id"],
        )
        members = self.backend.list_course_members(admin, course["id"])
        self.assertNotIn(
            student_registration["user"]["id"],
            {member["id"] for member in members},
        )

        reset = self.backend.request_password_reset({
            "email": "invited-professor@example.edu",
        })
        self.backend.confirm_password_reset({
            "token": reset["devPasswordResetToken"],
            "password": "new-professor-password-2026",
        })
        with self.assertRaises(ApiError):
            self.backend.login({
                "email": "invited-professor@example.edu",
                "password": "invited-professor-password",
            })
        login = self.backend.login({
            "email": "invited-professor@example.edu",
            "password": "new-professor-password-2026",
        })
        self.assertEqual(login["user"]["role"], "professor")

        provider = self.backend.create_identity_provider(admin, {
            "kind": "oidc",
            "name": "University SSO",
            "issuer": "https://idp.example.edu",
            "clientId": "neural-blocks-client",
            "authorizationEndpoint": "https://idp.example.edu/authorize",
            "tokenEndpoint": "https://idp.example.edu/token",
            "jwksUri": "https://idp.example.edu/jwks",
            "clientSecretEnv": "NBL_TEST_OIDC_SECRET",
            "defaultRole": "student",
        })
        self.assertEqual(provider["kind"], "oidc")
        public_providers = self.backend.public_identity_providers(
            "university-operations"
        )
        self.assertEqual(public_providers[0]["id"], provider["id"])
        provider_row = self.backend.get_identity_provider(provider_id=provider["id"])
        state = self.backend.create_federation_state(provider_row, "oidc")
        consumed = self.backend.consume_federation_state(state["state"], "oidc")
        self.assertEqual(consumed["nonce"], state["nonce"])
        with self.assertRaises(ApiError):
            self.backend.consume_federation_state(state["state"], "oidc")

        event_types = {
            event["eventType"]
            for event in self.backend.list_audit_events(admin)
        }
        self.assertIn("invitation.created", event_types)
        self.assertIn("course.member_removed", event_types)
        self.assertIn("identity_provider.created", event_types)
        self.assertGreaterEqual(len(self.mailer.messages), 4)

    def test_lti_roster_grade_passback_plan_and_provider_update(self):
        registration = self.register_institution("lti")
        admin = self.backend.authenticate(registration["sessionToken"])
        provider_payload = self.backend.create_identity_provider(admin, {
            "kind": "lti",
            "name": "University LMS",
            "issuer": "https://lms.example.edu",
            "clientId": "tool-client",
            "authorizationEndpoint": "https://lms.example.edu/authorize",
            "tokenEndpoint": "https://lms.example.edu/token",
            "jwksUri": "https://lms.example.edu/jwks",
            "clientSecretEnv": "NBL_TEST_LTI_SECRET",
            "serviceTokenAuthMethod": "client_secret_basic",
            "deploymentId": "deployment-1",
            "defaultRole": "student",
        })
        provider = self.backend.get_identity_provider(provider_id=provider_payload["id"])
        context_claim = {
            "https://purl.imsglobal.org/spec/lti/claim/context": {
                "id": "context-1",
                "label": "AI101",
                "title": "AI Foundations",
            },
            "https://purl.imsglobal.org/spec/lti/claim/resource_link": {
                "id": "resource-link-1",
            },
            "https://purl.imsglobal.org/spec/lti-nrps/claim/namesroleservice": {
                "context_memberships_url": "https://lms.example.edu/nrps/context-1",
                "service_versions": ["2.0"],
            },
            "https://purl.imsglobal.org/spec/lti-ags/claim/endpoint": {
                "scope": [
                    "https://purl.imsglobal.org/spec/lti-ags/scope/lineitem",
                    "https://purl.imsglobal.org/spec/lti-ags/scope/score",
                ],
                "lineitems": "https://lms.example.edu/ags/lineitems",
            },
        }
        professor_login = self.backend.resolve_federated_login(provider, {
            "sub": "lms-professor",
            "email": "lms-professor@example.edu",
            "name": "LMS Professor",
            "https://purl.imsglobal.org/spec/lti/claim/roles": [
                "http://purl.imsglobal.org/vocab/lis/v2/membership#Instructor",
            ],
            **context_claim,
        })
        course_id = professor_login["courseId"]
        professor = self.backend.authenticate(professor_login["sessionToken"])
        service = self.backend.get_lti_course_service(admin, course_id)
        self.assertTrue(service["connected"])
        self.assertTrue(service["nrps"]["available"])
        self.assertTrue(service["ags"]["canCreateLineItems"])
        self.assertTrue(service["ags"]["canSendScores"])

        sync = self.backend.apply_lti_roster(
            admin,
            course_id,
            provider["id"],
            "context-1",
            [
                {
                    "user_id": "lms-professor",
                    "email": "lms-professor@example.edu",
                    "name": "LMS Professor",
                    "status": "Active",
                    "roles": [
                        "http://purl.imsglobal.org/vocab/lis/v2/membership#Instructor",
                    ],
                },
                {
                    "user_id": "lms-student",
                    "email": "lms-student@example.edu",
                    "name": "LMS Student",
                    "status": "Active",
                    "roles": [
                        "http://purl.imsglobal.org/vocab/lis/v2/membership#Learner",
                    ],
                },
            ],
        )
        self.assertEqual(sync["received"], 2)
        self.assertEqual(sync["enrolled"], 2)
        self.assertEqual(len(self.backend.list_course_members(admin, course_id)), 2)

        student_login = self.backend.resolve_federated_login(provider, {
            "sub": "lms-student",
            "email": "lms-student@example.edu",
            "name": "LMS Student",
            "https://purl.imsglobal.org/spec/lti/claim/roles": [
                "http://purl.imsglobal.org/vocab/lis/v2/membership#Learner",
            ],
            "https://purl.imsglobal.org/spec/lti/claim/context": {
                "id": "context-1",
            },
        })
        student = self.backend.authenticate(student_login["sessionToken"])
        preserved = self.backend.get_lti_course_service(admin, course_id)
        self.assertTrue(preserved["nrps"]["available"])
        self.assertTrue(preserved["ags"]["canSendScores"])

        assignment = self.backend.create_assignment(professor, course_id, {
            "title": "LMS XOR",
            "instructions": "Train XOR.",
            "requiredFamily": "mlp",
            "targetAccuracy": 0.8,
        })
        snapshot = {
            "model": {"family": "mlp", "parameterCount": 33},
            "result": {"validationAccuracy": 0.93, "validationLoss": 0.15},
        }
        project = self.backend.save_project(student, course_id, {
            "name": "LMS Student Project",
            "snapshot": snapshot,
        })
        submission = self.backend.submit_assignment(student, assignment["id"], {
            "projectId": project["id"],
            "snapshot": snapshot,
        })
        self.backend.grade_submission(professor, submission["id"], {
            "score": 96,
            "feedback": "Passed",
        })
        plan = self.backend.prepare_lti_grade_passback(professor, submission["id"])
        self.assertEqual(plan["studentSubject"], "lms-student")
        self.assertEqual(plan["score"], 96)
        lineitem_url = self.backend.save_lti_line_item(
            professor,
            plan,
            "https://lms.example.edu/ags/lineitems/assignment-1",
        )
        passback = self.backend.record_lti_grade_passback(
            professor,
            plan,
            lineitem_url,
            {"response": {"status": "accepted"}},
        )
        self.assertEqual(passback["status"], "sent")

        updated = self.backend.update_identity_provider(admin, provider["id"], {
            "enabled": False,
            "name": "University LMS Disabled",
        })
        self.assertFalse(updated["enabled"])
        self.assertEqual(updated["name"], "University LMS Disabled")
        with self.assertRaises(ApiError):
            self.backend.get_identity_provider(provider_id=provider["id"])

        events = {
            event["eventType"]
            for event in self.backend.list_audit_events(admin)
        }
        self.assertIn("lti.roster_synced", events)
        self.assertIn("lti.grade_passback_sent", events)
        self.assertIn("identity_provider.updated", events)


if __name__ == "__main__":
    unittest.main()
