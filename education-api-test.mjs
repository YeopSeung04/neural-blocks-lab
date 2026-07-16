import assert from "node:assert/strict";
import { EducationApi, EducationApiError } from "./education-api.mjs";

function response(status, payload) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(payload),
  };
}

const requests = [];
const queue = [
  response(200, {
    user: { id: "user_1", displayName: "Admin", role: "admin" },
    tenant: { id: "tenant_1", name: "Test University" },
    csrfToken: "csrf-token",
  }),
  response(200, { status: "verified" }),
  response(201, {
    course: { id: "course_1", code: "AI101", joinCode: "JOIN123" },
  }),
  response(200, {
    members: [{ id: "user_1", displayName: "Admin", courseRole: "instructor" }],
  }),
  response(201, {
    invitation: { id: "invite_1", email: "professor@example.edu" },
  }),
  response(403, {
    error: { code: "forbidden", message: "Denied" },
  }),
];

const api = new EducationApi(async (path, options) => {
  requests.push({ path, options });
  return queue.shift();
});

await api.me();
await api.verifyEmail("verification-token");
await api.createCourse({ name: "AI", code: "AI101", term: "2026-2" });
await api.listCourseMembers("course_1");
await api.createInvitation({
  email: "professor@example.edu",
  role: "professor",
  courseId: "course_1",
});
assert.equal(requests[1].options.headers["X-CSRF-Token"], undefined);
assert.equal(requests[2].options.headers["X-CSRF-Token"], "csrf-token");
assert.equal(requests[4].options.headers["X-CSRF-Token"], "csrf-token");
assert.equal(requests[2].options.credentials, "same-origin");

await assert.rejects(
  () => api.joinCourse("WRONG"),
  (error) => {
    assert.ok(error instanceof EducationApiError);
    assert.equal(error.status, 403);
    assert.equal(error.code, "forbidden");
    assert.equal(error.message, "Denied");
    return true;
  },
);

console.log("Education API client tests passed");
