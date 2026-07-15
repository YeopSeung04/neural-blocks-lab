import { EducationStore } from "./education-store.mjs";

function formatDate(value) {
  if (!value) return "마감 없음";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "날짜 오류";
  return new Intl.DateTimeFormat("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function metricText(snapshot) {
  const accuracy = Number(snapshot?.result?.validationAccuracy);
  const loss = Number(snapshot?.result?.validationLoss);
  const parts = [];
  if (Number.isFinite(accuracy)) parts.push(`val acc ${(accuracy * 100).toFixed(1)}%`);
  if (Number.isFinite(loss)) parts.push(`val loss ${loss.toFixed(4)}`);
  return parts.join(" · ") || "학습 결과 없음";
}

function createEmpty(message) {
  const empty = document.createElement("div");
  empty.className = "education-empty";
  empty.textContent = message;
  return empty;
}

function createMeta(values) {
  const meta = document.createElement("div");
  meta.className = "education-meta";
  for (const value of values.filter(Boolean)) {
    const item = document.createElement("span");
    item.textContent = value;
    meta.append(item);
  }
  return meta;
}

function downloadJson(filename, text) {
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function mountEducationWorkspace({
  captureExperiment,
  restoreExperiment,
  store = new EducationStore(),
}) {
  const elements = {
    open: document.getElementById("educationOpenButton"),
    dialog: document.getElementById("educationDialog"),
    close: document.getElementById("educationCloseButton"),
    role: document.getElementById("educationRole"),
    actorName: document.getElementById("educationActorName"),
    courseSelect: document.getElementById("educationCourseSelect"),
    status: document.getElementById("educationStatus"),
    professorTools: document.getElementById("educationProfessorTools"),
    studentTools: document.getElementById("educationStudentTools"),
    courseForm: document.getElementById("courseForm"),
    courseName: document.getElementById("courseName"),
    courseCode: document.getElementById("courseCode"),
    courseTerm: document.getElementById("courseTerm"),
    assignmentForm: document.getElementById("assignmentForm"),
    assignmentTitle: document.getElementById("assignmentTitle"),
    assignmentInstructions: document.getElementById("assignmentInstructions"),
    assignmentDueAt: document.getElementById("assignmentDueAt"),
    assignmentFamily: document.getElementById("assignmentFamily"),
    assignmentTargetAccuracy: document.getElementById("assignmentTargetAccuracy"),
    assignmentIncludeStarter: document.getElementById("assignmentIncludeStarter"),
    studentProjectName: document.getElementById("studentProjectName"),
    studentAssignmentSelect: document.getElementById("studentAssignmentSelect"),
    saveProject: document.getElementById("saveCurrentProjectButton"),
    loadStarter: document.getElementById("loadAssignmentStarterButton"),
    submit: document.getElementById("submitCurrentExperimentButton"),
    assignmentList: document.getElementById("educationAssignmentsList"),
    projectList: document.getElementById("educationProjectsList"),
    submissionList: document.getElementById("educationSubmissionsList"),
    assignmentCount: document.getElementById("assignmentCount"),
    projectCount: document.getElementById("projectCount"),
    submissionCount: document.getElementById("submissionCount"),
    exportData: document.getElementById("exportEducationDataButton"),
    resetData: document.getElementById("resetEducationDataButton"),
    importData: document.getElementById("importEducationDataInput"),
  };

  let activeProjectId = null;

  function actor() {
    const role = elements.role.value;
    const name = elements.actorName.value.trim() || (role === "professor" ? "김교수" : "이학생");
    return {
      id: `${role}:${name.toLowerCase().replace(/\s+/g, "-")}`,
      name,
      role,
    };
  }

  function currentCourseId() {
    return elements.courseSelect.value || null;
  }

  function setStatus(message, state = "neutral") {
    elements.status.className = "education-status";
    if (state === "error" || state === "success") elements.status.classList.add(state);
    elements.status.textContent = message;
  }

  function populateCourseSelect(preferredId = null) {
    const courses = store.listCourses();
    const current = preferredId ?? elements.courseSelect.value;
    elements.courseSelect.replaceChildren();
    for (const course of courses) {
      const option = document.createElement("option");
      option.value = course.id;
      option.textContent = `${course.code} · ${course.name} · ${course.term}`;
      elements.courseSelect.append(option);
    }
    if (courses.some((course) => course.id === current)) {
      elements.courseSelect.value = current;
    }
  }

  function populateStudentAssignmentSelect(assignments) {
    const current = elements.studentAssignmentSelect.value;
    elements.studentAssignmentSelect.replaceChildren();
    for (const assignment of assignments) {
      const option = document.createElement("option");
      option.value = assignment.id;
      option.textContent = assignment.title;
      elements.studentAssignmentSelect.append(option);
    }
    if (assignments.some((assignment) => assignment.id === current)) {
      elements.studentAssignmentSelect.value = current;
    }
  }

  function renderAssignments(assignments) {
    elements.assignmentList.replaceChildren();
    elements.assignmentCount.textContent = String(assignments.length);
    if (!assignments.length) {
      elements.assignmentList.append(createEmpty("이 강좌에 등록된 과제가 없습니다."));
      return;
    }
    for (const assignment of assignments) {
      const item = document.createElement("article");
      item.className = "education-list-item assignment";
      const title = document.createElement("strong");
      title.textContent = assignment.title;
      const description = document.createElement("p");
      description.textContent = assignment.instructions || "과제 설명이 없습니다.";
      item.append(
        title,
        description,
        createMeta([
          assignment.requiredFamily === "any" ? "모델 자유" : assignment.requiredFamily.toUpperCase(),
          assignment.targetAccuracy === null
            ? null
            : `목표 ${(assignment.targetAccuracy * 100).toFixed(0)}%`,
          formatDate(assignment.dueAt),
          assignment.starterSnapshot ? "시작 템플릿 포함" : null,
        ]),
      );
      if (elements.role.value === "student") {
        const actions = document.createElement("div");
        actions.className = "education-item-actions";
        const selectButton = document.createElement("button");
        selectButton.type = "button";
        selectButton.dataset.action = "select-assignment";
        selectButton.dataset.assignmentId = assignment.id;
        selectButton.textContent = "제출 과제로 선택";
        const loadButton = document.createElement("button");
        loadButton.type = "button";
        loadButton.dataset.action = "load-starter";
        loadButton.dataset.assignmentId = assignment.id;
        loadButton.textContent = "시작 상태 불러오기";
        loadButton.disabled = !assignment.starterSnapshot;
        actions.append(selectButton, loadButton);
        item.append(actions);
      }
      elements.assignmentList.append(item);
    }
  }

  function renderProjects(projects) {
    elements.projectList.replaceChildren();
    elements.projectCount.textContent = String(projects.length);
    if (!projects.length) {
      elements.projectList.append(createEmpty("저장된 실험 프로젝트가 없습니다."));
      return;
    }
    for (const project of projects) {
      const latest = project.versions.find((version) => version.id === project.latestVersionId);
      const item = document.createElement("article");
      item.className = "education-list-item project";
      const title = document.createElement("strong");
      title.textContent = project.name;
      const description = document.createElement("p");
      description.textContent =
        `${project.ownerName} · ${project.versions.length} versions · ${metricText(latest?.snapshot)}`;
      const actions = document.createElement("div");
      actions.className = "education-item-actions";
      const loadButton = document.createElement("button");
      loadButton.type = "button";
      loadButton.dataset.action = "load-project";
      loadButton.dataset.projectId = project.id;
      loadButton.textContent = "실험 불러오기";
      actions.append(loadButton);
      item.append(
        title,
        description,
        createMeta([
          latest?.snapshot?.model?.family?.toUpperCase(),
          `updated ${formatDate(project.updatedAt)}`,
        ]),
        actions,
      );
      elements.projectList.append(item);
    }
  }

  function renderSubmissions(submissions) {
    elements.submissionList.replaceChildren();
    elements.submissionCount.textContent = String(submissions.length);
    if (!submissions.length) {
      elements.submissionList.append(createEmpty("제출된 실험이 없습니다."));
      return;
    }
    const assignments = new Map(store.listAssignments().map((item) => [item.id, item]));
    for (const submission of submissions.slice().reverse()) {
      const assignment = assignments.get(submission.assignmentId);
      const item = document.createElement("article");
      item.className = "education-list-item submission";
      const title = document.createElement("strong");
      title.textContent = `${assignment?.title ?? "삭제된 과제"} · ${submission.studentName}`;
      const description = document.createElement("p");
      const auto = submission.autoEvaluation?.passed ? "자동 조건 통과" : "자동 조건 미통과";
      description.textContent =
        `Attempt ${submission.attempt} · ${metricText(submission.snapshot)} · ${auto}`;
      item.append(
        title,
        description,
        createMeta([
          submission.status === "graded" ? `점수 ${submission.score}` : "채점 대기",
          formatDate(submission.submittedAt),
        ]),
      );
      if (elements.role.value === "professor") {
        const grade = document.createElement("div");
        grade.className = "education-grade-grid";
        const score = document.createElement("input");
        score.type = "number";
        score.min = "0";
        score.max = "100";
        score.step = "0.5";
        score.value = submission.score ?? submission.autoEvaluation?.suggestedScore ?? "";
        score.placeholder = "점수";
        score.dataset.gradeScore = submission.id;
        const gradeButton = document.createElement("button");
        gradeButton.type = "button";
        gradeButton.className = "primary";
        gradeButton.dataset.action = "grade-submission";
        gradeButton.dataset.submissionId = submission.id;
        gradeButton.textContent = "채점 저장";
        const feedback = document.createElement("textarea");
        feedback.rows = 2;
        feedback.value = submission.feedback ?? "";
        feedback.placeholder = "교수 피드백";
        feedback.dataset.gradeFeedback = submission.id;
        grade.append(score, gradeButton, feedback);
        item.append(grade);
      } else if (submission.feedback) {
        const feedback = document.createElement("p");
        feedback.textContent = `교수 피드백: ${submission.feedback}`;
        item.append(feedback);
      }
      elements.submissionList.append(item);
    }
  }

  function render() {
    const courseId = currentCourseId();
    const currentActor = actor();
    const assignments = courseId ? store.listAssignments(courseId) : [];
    const projects = elements.role.value === "professor"
      ? store.listProjects(null, courseId)
      : store.listProjects(currentActor.id, courseId);
    const submissions = elements.role.value === "professor"
      ? store.listSubmissions({ courseId })
      : store.listSubmissions({ courseId, studentId: currentActor.id });
    elements.professorTools.hidden = elements.role.value !== "professor";
    elements.studentTools.hidden = elements.role.value !== "student";
    populateStudentAssignmentSelect(assignments);
    renderAssignments(assignments);
    renderProjects(projects);
    renderSubmissions(submissions);
  }

  function saveCurrentProject() {
    const courseId = currentCourseId();
    if (!courseId) throw new Error("강좌를 먼저 선택하세요.");
    const currentActor = actor();
    const name = elements.studentProjectName.value.trim() || "나의 AI 실험";
    if (!activeProjectId) {
      const existing = store.listProjects(currentActor.id, courseId)
        .find((project) => project.name === name);
      activeProjectId = existing?.id ?? null;
    }
    const project = store.saveProject({
      projectId: activeProjectId,
      courseId,
      name,
      owner: currentActor,
      snapshot: captureExperiment(),
    });
    activeProjectId = project.id;
    return project;
  }

  function loadAssignmentStarter(assignmentId) {
    const assignment = store.listAssignments().find((item) => item.id === assignmentId);
    if (!assignment?.starterSnapshot) throw new Error("이 과제에는 시작 템플릿이 없습니다.");
    const message = restoreExperiment(assignment.starterSnapshot);
    elements.dialog.close();
    setStatus(message || `${assignment.title} 시작 상태를 불러왔습니다.`, "success");
  }

  const demoProfessor = { id: "professor:김교수", name: "김교수", role: "professor" };
  const demoCourse = store.ensureDemoCourse(demoProfessor);
  populateCourseSelect(demoCourse.id);
  render();

  elements.open.addEventListener("click", () => {
    populateCourseSelect();
    render();
    elements.dialog.showModal();
  });
  elements.close.addEventListener("click", () => elements.dialog.close());
  elements.dialog.addEventListener("click", (event) => {
    if (event.target === elements.dialog) elements.dialog.close();
  });
  elements.role.addEventListener("change", () => {
    elements.actorName.value = elements.role.value === "professor" ? "김교수" : "이학생";
    activeProjectId = null;
    render();
  });
  elements.actorName.addEventListener("change", () => {
    activeProjectId = null;
    render();
  });
  elements.courseSelect.addEventListener("change", () => {
    activeProjectId = null;
    render();
  });

  elements.courseForm.addEventListener("submit", (event) => {
    event.preventDefault();
    try {
      const course = store.createCourse({
        name: elements.courseName.value,
        code: elements.courseCode.value,
        term: elements.courseTerm.value,
        owner: actor(),
      });
      elements.courseForm.reset();
      populateCourseSelect(course.id);
      render();
      setStatus(`${course.code} 강좌를 생성했습니다.`, "success");
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  elements.assignmentForm.addEventListener("submit", (event) => {
    event.preventDefault();
    try {
      const targetPercent = elements.assignmentTargetAccuracy.value;
      const assignment = store.createAssignment({
        courseId: currentCourseId(),
        title: elements.assignmentTitle.value,
        instructions: elements.assignmentInstructions.value,
        dueAt: elements.assignmentDueAt.value,
        requiredFamily: elements.assignmentFamily.value,
        targetAccuracy: targetPercent === "" ? null : Number(targetPercent) / 100,
        starterSnapshot: elements.assignmentIncludeStarter.checked
          ? captureExperiment()
          : null,
        createdBy: actor(),
      });
      elements.assignmentForm.reset();
      elements.assignmentIncludeStarter.checked = true;
      render();
      setStatus(`${assignment.title} 과제를 생성했습니다.`, "success");
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  elements.saveProject.addEventListener("click", () => {
    try {
      const project = saveCurrentProject();
      render();
      setStatus(`${project.name}의 ${project.versions.length}번째 버전을 저장했습니다.`, "success");
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  elements.loadStarter.addEventListener("click", () => {
    try {
      loadAssignmentStarter(elements.studentAssignmentSelect.value);
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  elements.submit.addEventListener("click", () => {
    try {
      const assignmentId = elements.studentAssignmentSelect.value;
      if (!assignmentId) throw new Error("제출할 과제를 선택하세요.");
      const project = saveCurrentProject();
      const snapshot = store.latestProjectSnapshot(project.id);
      const submission = store.submitAssignment({
        assignmentId,
        projectId: project.id,
        student: actor(),
        snapshot,
      });
      render();
      setStatus(
        `${submission.studentName}의 ${submission.attempt}번째 제출을 저장했습니다.`,
        "success",
      );
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  elements.assignmentList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-action]");
    if (!button) return;
    try {
      if (button.dataset.action === "select-assignment") {
        elements.studentAssignmentSelect.value = button.dataset.assignmentId;
        setStatus("제출할 과제를 선택했습니다.", "success");
      }
      if (button.dataset.action === "load-starter") {
        loadAssignmentStarter(button.dataset.assignmentId);
      }
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  elements.projectList.addEventListener("click", (event) => {
    const button = event.target.closest('[data-action="load-project"]');
    if (!button) return;
    try {
      const projects = store.listProjects();
      const project = projects.find((item) => item.id === button.dataset.projectId);
      const snapshot = store.latestProjectSnapshot(button.dataset.projectId);
      activeProjectId = project.id;
      elements.studentProjectName.value = project.name;
      const message = restoreExperiment(snapshot);
      elements.dialog.close();
      setStatus(message || `${project.name} 프로젝트를 불러왔습니다.`, "success");
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  elements.submissionList.addEventListener("click", (event) => {
    const button = event.target.closest('[data-action="grade-submission"]');
    if (!button) return;
    const submissionId = button.dataset.submissionId;
    try {
      const score = elements.submissionList
        .querySelector(`[data-grade-score="${submissionId}"]`).value;
      const feedback = elements.submissionList
        .querySelector(`[data-grade-feedback="${submissionId}"]`).value;
      const submission = store.gradeSubmission({
        submissionId,
        score,
        feedback,
        grader: actor(),
      });
      render();
      setStatus(`${submission.studentName} 제출을 ${submission.score}점으로 채점했습니다.`, "success");
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  elements.exportData.addEventListener("click", () => {
    downloadJson("neural-blocks-classroom.json", store.exportJson());
    setStatus("수업 데이터를 JSON으로 내보냈습니다.", "success");
  });

  elements.resetData.addEventListener("click", () => {
    store.reset();
    const course = store.ensureDemoCourse(demoProfessor);
    activeProjectId = null;
    elements.role.value = "professor";
    elements.actorName.value = "김교수";
    populateCourseSelect(course.id);
    render();
    setStatus("교육 데모 데이터를 기본 강좌 상태로 초기화했습니다.", "success");
  });

  elements.importData.addEventListener("change", async () => {
    const file = elements.importData.files[0];
    if (!file) return;
    try {
      store.importJson(await file.text());
      populateCourseSelect();
      render();
      setStatus("수업 데이터 JSON을 가져왔습니다.", "success");
    } catch (error) {
      setStatus(`가져오기 실패: ${error.message}`, "error");
    } finally {
      elements.importData.value = "";
    }
  });

  return {
    store,
    open: () => {
      populateCourseSelect();
      render();
      elements.dialog.showModal();
    },
  };
}
