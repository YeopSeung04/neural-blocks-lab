export const EDUCATION_SCHEMA_VERSION = 1;

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function createEmptyState() {
  return {
    schemaVersion: EDUCATION_SCHEMA_VERSION,
    courses: [],
    assignments: [],
    projects: [],
    submissions: [],
  };
}

function requiredText(value, label) {
  const text = String(value ?? "").trim();
  if (!text) throw new Error(`${label} is required`);
  return text;
}

function normalizeScore(value) {
  if (value === null || value === undefined || value === "") return null;
  const score = Number(value);
  if (!Number.isFinite(score) || score < 0 || score > 100) {
    throw new Error("Score must be between 0 and 100");
  }
  return Math.round(score * 10) / 10;
}

function validateState(state) {
  if (!state || state.schemaVersion !== EDUCATION_SCHEMA_VERSION) {
    throw new Error(`Unsupported education data version: ${state?.schemaVersion ?? "unknown"}`);
  }
  for (const collection of ["courses", "assignments", "projects", "submissions"]) {
    if (!Array.isArray(state[collection])) {
      throw new Error(`Education data is missing ${collection}`);
    }
  }
  return clone(state);
}

export class MemoryStorage {
  constructor() {
    this.values = new Map();
  }

  getItem(key) {
    return this.values.has(key) ? this.values.get(key) : null;
  }

  setItem(key, value) {
    this.values.set(key, String(value));
  }

  removeItem(key) {
    this.values.delete(key);
  }
}

export class EducationStore {
  constructor({
    storage = globalThis.localStorage,
    storageKey = "neural-blocks-education-v1",
    now = () => new Date().toISOString(),
    idFactory = (prefix) => `${prefix}_${crypto.randomUUID()}`,
  } = {}) {
    this.storage = storage;
    this.storageKey = storageKey;
    this.now = now;
    this.idFactory = idFactory;
    this.state = this.load();
  }

  load() {
    const raw = this.storage?.getItem?.(this.storageKey);
    if (!raw) return createEmptyState();
    try {
      return validateState(JSON.parse(raw));
    } catch {
      return createEmptyState();
    }
  }

  persist() {
    this.storage?.setItem?.(this.storageKey, JSON.stringify(this.state));
  }

  snapshot() {
    return clone(this.state);
  }

  exportJson() {
    return JSON.stringify(this.state, null, 2);
  }

  importJson(text) {
    this.state = validateState(JSON.parse(text));
    this.persist();
    return this.snapshot();
  }

  reset() {
    this.state = createEmptyState();
    this.persist();
    return this.snapshot();
  }

  createCourse({ name, code, term, owner }) {
    const actor = this.normalizeActor(owner, "professor");
    const course = {
      id: this.idFactory("course"),
      name: requiredText(name, "Course name"),
      code: requiredText(code, "Course code"),
      term: requiredText(term, "Term"),
      ownerId: actor.id,
      ownerName: actor.name,
      createdAt: this.now(),
    };
    this.state.courses.push(course);
    this.persist();
    return clone(course);
  }

  createAssignment({
    courseId,
    title,
    instructions = "",
    dueAt = "",
    requiredFamily = "any",
    targetAccuracy = null,
    starterSnapshot = null,
    createdBy,
  }) {
    const course = this.requireCourse(courseId);
    const actor = this.normalizeActor(createdBy, "professor");
    const accuracy = targetAccuracy === "" || targetAccuracy === null
      ? null
      : Number(targetAccuracy);
    if (accuracy !== null && (!Number.isFinite(accuracy) || accuracy < 0 || accuracy > 1)) {
      throw new Error("Target accuracy must be between 0 and 1");
    }
    const assignment = {
      id: this.idFactory("assignment"),
      courseId: course.id,
      title: requiredText(title, "Assignment title"),
      instructions: String(instructions ?? "").trim(),
      dueAt: dueAt ? new Date(dueAt).toISOString() : null,
      requiredFamily,
      targetAccuracy: accuracy,
      starterSnapshot: starterSnapshot ? clone(starterSnapshot) : null,
      createdBy: actor.id,
      createdByName: actor.name,
      createdAt: this.now(),
    };
    this.state.assignments.push(assignment);
    this.persist();
    return clone(assignment);
  }

  saveProject({ projectId = null, courseId = null, name, owner, snapshot }) {
    const actor = this.normalizeActor(owner);
    if (courseId) this.requireCourse(courseId);
    const version = {
      id: this.idFactory("version"),
      savedAt: this.now(),
      snapshot: clone(snapshot),
    };
    let project = projectId
      ? this.state.projects.find((item) => item.id === projectId)
      : null;
    if (project && project.ownerId !== actor.id) {
      throw new Error("Only the project owner can save a new version");
    }
    if (!project) {
      project = {
        id: this.idFactory("project"),
        courseId,
        name: requiredText(name, "Project name"),
        ownerId: actor.id,
        ownerName: actor.name,
        createdAt: this.now(),
        updatedAt: this.now(),
        versions: [],
      };
      this.state.projects.push(project);
    }
    project.name = requiredText(name, "Project name");
    project.courseId = courseId;
    project.updatedAt = this.now();
    project.versions.push(version);
    project.latestVersionId = version.id;
    this.persist();
    return clone(project);
  }

  submitAssignment({ assignmentId, projectId = null, student, snapshot }) {
    const assignment = this.requireAssignment(assignmentId);
    const actor = this.normalizeActor(student, "student");
    if (projectId) {
      const project = this.requireProject(projectId);
      if (project.ownerId !== actor.id) {
        throw new Error("Only the project owner can submit it");
      }
    }
    const previousAttempts = this.state.submissions.filter((submission) =>
      submission.assignmentId === assignmentId && submission.studentId === actor.id);
    const autoEvaluation = this.evaluateAssignment(assignment, snapshot);
    const submission = {
      id: this.idFactory("submission"),
      assignmentId,
      courseId: assignment.courseId,
      projectId,
      studentId: actor.id,
      studentName: actor.name,
      attempt: previousAttempts.length + 1,
      submittedAt: this.now(),
      status: "submitted",
      snapshot: clone(snapshot),
      autoEvaluation,
      score: null,
      feedback: "",
      gradedAt: null,
      gradedBy: null,
    };
    this.state.submissions.push(submission);
    this.persist();
    return clone(submission);
  }

  evaluateAssignment(assignment, snapshot) {
    const family = snapshot?.model?.family ?? snapshot?.family ?? "unknown";
    const validationAccuracy = Number(
      snapshot?.result?.validationAccuracy ??
      snapshot?.metrics?.validationAccuracy,
    );
    const checks = [];
    if (assignment.requiredFamily !== "any") {
      checks.push({
        label: "Model family",
        passed: family === assignment.requiredFamily,
        detail: `${family} / required ${assignment.requiredFamily}`,
      });
    }
    if (assignment.targetAccuracy !== null) {
      checks.push({
        label: "Validation accuracy",
        passed:
          Number.isFinite(validationAccuracy) &&
          validationAccuracy >= assignment.targetAccuracy,
        detail: Number.isFinite(validationAccuracy)
          ? `${(validationAccuracy * 100).toFixed(1)}% / target ${(assignment.targetAccuracy * 100).toFixed(1)}%`
          : "No validation accuracy",
      });
    }
    const passed = checks.every((check) => check.passed);
    return {
      passed,
      checks,
      suggestedScore: checks.length
        ? Math.round(checks.filter((check) => check.passed).length / checks.length * 100)
        : null,
    };
  }

  gradeSubmission({ submissionId, score, feedback = "", grader }) {
    const submission = this.requireSubmission(submissionId);
    const actor = this.normalizeActor(grader, "professor");
    submission.score = normalizeScore(score);
    submission.feedback = String(feedback ?? "").trim();
    submission.status = "graded";
    submission.gradedAt = this.now();
    submission.gradedBy = actor.id;
    submission.gradedByName = actor.name;
    this.persist();
    return clone(submission);
  }

  listCourses() {
    return clone(this.state.courses);
  }

  listAssignments(courseId = null) {
    return clone(this.state.assignments.filter((assignment) =>
      !courseId || assignment.courseId === courseId));
  }

  listProjects(ownerId = null, courseId = null) {
    return clone(this.state.projects.filter((project) =>
      (!ownerId || project.ownerId === ownerId) &&
      (!courseId || project.courseId === courseId)));
  }

  listSubmissions({ courseId = null, assignmentId = null, studentId = null } = {}) {
    return clone(this.state.submissions.filter((submission) =>
      (!courseId || submission.courseId === courseId) &&
      (!assignmentId || submission.assignmentId === assignmentId) &&
      (!studentId || submission.studentId === studentId)));
  }

  latestProjectSnapshot(projectId) {
    const project = this.requireProject(projectId);
    const version = project.versions.find((item) => item.id === project.latestVersionId);
    return version ? clone(version.snapshot) : null;
  }

  ensureDemoCourse(owner) {
    if (this.state.courses.length) return clone(this.state.courses[0]);
    const course = this.createCourse({
      name: "AI 기초 실습",
      code: "AI101",
      term: "2026-2",
      owner,
    });
    this.createAssignment({
      courseId: course.id,
      title: "XOR 분류 모델 만들기",
      instructions: "Dense 블록과 Adam을 사용해 validation accuracy 80% 이상을 달성하세요.",
      requiredFamily: "mlp",
      targetAccuracy: 0.8,
      createdBy: owner,
    });
    return course;
  }

  normalizeActor(actor, requiredRole = null) {
    const role = requiredRole ?? actor?.role ?? "student";
    if (requiredRole && actor?.role && actor.role !== requiredRole) {
      throw new Error(`${requiredRole} role is required`);
    }
    const name = requiredText(actor?.name, "Actor name");
    return {
      id: actor?.id || `${role}:${name.toLowerCase().replace(/\s+/g, "-")}`,
      name,
      role,
    };
  }

  requireCourse(courseId) {
    const course = this.state.courses.find((item) => item.id === courseId);
    if (!course) throw new Error("Course not found");
    return course;
  }

  requireAssignment(assignmentId) {
    const assignment = this.state.assignments.find((item) => item.id === assignmentId);
    if (!assignment) throw new Error("Assignment not found");
    return assignment;
  }

  requireProject(projectId) {
    const project = this.state.projects.find((item) => item.id === projectId);
    if (!project) throw new Error("Project not found");
    return project;
  }

  requireSubmission(submissionId) {
    const submission = this.state.submissions.find((item) => item.id === submissionId);
    if (!submission) throw new Error("Submission not found");
    return submission;
  }
}
