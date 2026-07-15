import assert from "node:assert/strict";
import {
  EducationStore,
  MemoryStorage,
} from "./education-store.mjs";

let id = 0;
let tick = 0;
const storage = new MemoryStorage();
const store = new EducationStore({
  storage,
  now: () => `2026-07-15T00:00:${String(tick++).padStart(2, "0")}.000Z`,
  idFactory: (prefix) => `${prefix}_${++id}`,
});
const professor = { id: "professor:kim", name: "김교수", role: "professor" };
const student = { id: "student:lee", name: "이학생", role: "student" };

const course = store.createCourse({
  name: "AI 기초",
  code: "AI101",
  term: "2026-2",
  owner: professor,
});
const starterSnapshot = {
  model: { family: "mlp" },
  result: { validationAccuracy: 0.5 },
};
const assignment = store.createAssignment({
  courseId: course.id,
  title: "XOR 실습",
  requiredFamily: "mlp",
  targetAccuracy: 0.8,
  starterSnapshot,
  createdBy: professor,
});
assert.equal(store.listAssignments(course.id).length, 1);

const projectV1 = store.saveProject({
  courseId: course.id,
  name: "XOR 프로젝트",
  owner: student,
  snapshot: starterSnapshot,
});
const projectV2 = store.saveProject({
  projectId: projectV1.id,
  courseId: course.id,
  name: "XOR 프로젝트",
  owner: student,
  snapshot: {
    model: { family: "mlp" },
    result: { validationAccuracy: 0.92 },
  },
});
assert.equal(projectV2.versions.length, 2);
assert.equal(store.latestProjectSnapshot(projectV2.id).result.validationAccuracy, 0.92);

const submission = store.submitAssignment({
  assignmentId: assignment.id,
  projectId: projectV2.id,
  student,
  snapshot: store.latestProjectSnapshot(projectV2.id),
});
assert.equal(submission.attempt, 1);
assert.equal(submission.autoEvaluation.passed, true);
assert.equal(submission.autoEvaluation.suggestedScore, 100);

const graded = store.gradeSubmission({
  submissionId: submission.id,
  score: 95,
  feedback: "구조와 결과가 명확합니다.",
  grader: professor,
});
assert.equal(graded.status, "graded");
assert.equal(graded.score, 95);

const restored = new EducationStore({
  storage: new MemoryStorage(),
  idFactory: (prefix) => `${prefix}_restored`,
});
restored.importJson(store.exportJson());
assert.equal(restored.listCourses().length, 1);
assert.equal(restored.listProjects(student.id).length, 1);
assert.equal(restored.listSubmissions({ studentId: student.id })[0].score, 95);
restored.reset();
assert.equal(restored.listCourses().length, 0);

console.log(
  "education store: course, assignment, project versions, grading, import and reset passed",
);
