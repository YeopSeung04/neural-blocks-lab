import { EducationApi } from "./education-api.mjs";

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

function roleLabel(role) {
  return {
    admin: "대학 관리자",
    professor: "교수",
    student: "학생",
  }[role] || role;
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

export function mountEducationWorkspace({
  captureExperiment,
  restoreExperiment,
  api = new EducationApi(),
}) {
  const elements = {
    open: document.getElementById("educationOpenButton"),
    accountBadge: document.getElementById("accountBadge"),
    authDialog: document.getElementById("authDialog"),
    authClose: document.getElementById("authCloseButton"),
    authStatus: document.getElementById("authStatus"),
    loginForm: document.getElementById("loginForm"),
    loginEmail: document.getElementById("loginEmail"),
    loginPassword: document.getElementById("loginPassword"),
    registerForm: document.getElementById("registerForm"),
    registrationMode: document.getElementById("registrationMode"),
    registerDisplayName: document.getElementById("registerDisplayName"),
    registerEmail: document.getElementById("registerEmail"),
    registerPassword: document.getElementById("registerPassword"),
    institutionCreateFields: document.getElementById("institutionCreateFields"),
    institutionJoinFields: document.getElementById("institutionJoinFields"),
    institutionName: document.getElementById("institutionName"),
    institutionSlug: document.getElementById("institutionSlug"),
    institutionJoinCode: document.getElementById("institutionJoinCode"),
    dialog: document.getElementById("educationDialog"),
    close: document.getElementById("educationCloseButton"),
    logout: document.getElementById("educationLogoutButton"),
    userBadge: document.getElementById("educationUserBadge"),
    tenantBadge: document.getElementById("educationTenantBadge"),
    tenantCodePanel: document.getElementById("educationTenantCodePanel"),
    tenantJoinCode: document.getElementById("educationTenantJoinCode"),
    courseCodePanel: document.getElementById("educationCourseCodePanel"),
    courseJoinCodeDisplay: document.getElementById("educationCourseJoinCode"),
    courseSelect: document.getElementById("educationCourseSelect"),
    status: document.getElementById("educationStatus"),
    professorTools: document.getElementById("educationProfessorTools"),
    studentTools: document.getElementById("educationStudentTools"),
    courseForm: document.getElementById("courseForm"),
    courseName: document.getElementById("courseName"),
    courseCode: document.getElementById("courseCode"),
    courseTerm: document.getElementById("courseTerm"),
    courseJoinForm: document.getElementById("courseJoinForm"),
    courseJoinCode: document.getElementById("courseJoinCode"),
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
  };

  const state = {
    auth: null,
    courses: [],
    assignments: [],
    projects: [],
    submissions: [],
    activeProjectId: null,
  };
  let authReady = Promise.resolve();

  function isInstructor() {
    return ["admin", "professor"].includes(state.auth?.user?.role);
  }

  function isStudent() {
    return state.auth?.user?.role === "student";
  }

  function currentCourseId() {
    return elements.courseSelect.value || null;
  }

  function currentCourse() {
    return state.courses.find((course) => course.id === currentCourseId()) || null;
  }

  function setMessage(element, message, status = "neutral") {
    element.className = "education-status";
    if (status === "error" || status === "success") element.classList.add(status);
    element.textContent = message;
  }

  function setStatus(message, status = "neutral") {
    setMessage(elements.status, message, status);
  }

  function setAuthStatus(message, status = "neutral") {
    setMessage(elements.authStatus, message, status);
  }

  function updateRegistrationFields() {
    const createInstitution = elements.registrationMode.value === "create";
    elements.institutionCreateFields.hidden = !createInstitution;
    elements.institutionJoinFields.hidden = createInstitution;
    elements.institutionName.required = createInstitution;
    elements.institutionSlug.required = createInstitution;
    elements.institutionJoinCode.required = !createInstitution;
  }

  function updateIdentity() {
    if (!state.auth) {
      elements.accountBadge.textContent = "로그인 필요";
      elements.userBadge.textContent = "-";
      elements.tenantBadge.textContent = "-";
      elements.tenantCodePanel.hidden = true;
      return;
    }
    const { user, tenant } = state.auth;
    elements.accountBadge.textContent = `${user.displayName} · ${roleLabel(user.role)}`;
    elements.userBadge.textContent = `${user.displayName} · ${roleLabel(user.role)}`;
    elements.tenantBadge.textContent = `${tenant.name} / ${tenant.slug}`;
    elements.tenantCodePanel.hidden = !tenant.joinCode;
    elements.tenantJoinCode.textContent = tenant.joinCode || "-";
  }

  function populateCourseSelect(preferredId = null) {
    const previous = preferredId || currentCourseId();
    elements.courseSelect.replaceChildren();
    if (!state.courses.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = isStudent() ? "가입한 강좌 없음" : "생성된 강좌 없음";
      elements.courseSelect.append(option);
      elements.courseSelect.disabled = true;
      return;
    }
    elements.courseSelect.disabled = false;
    for (const course of state.courses) {
      const option = document.createElement("option");
      option.value = course.id;
      option.textContent = `${course.code} · ${course.name} · ${course.term}`;
      elements.courseSelect.append(option);
    }
    if (state.courses.some((course) => course.id === previous)) {
      elements.courseSelect.value = previous;
    }
  }

  function populateStudentAssignmentSelect() {
    const previous = elements.studentAssignmentSelect.value;
    elements.studentAssignmentSelect.replaceChildren();
    if (!state.assignments.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "제출 가능한 과제 없음";
      elements.studentAssignmentSelect.append(option);
      elements.studentAssignmentSelect.disabled = true;
      return;
    }
    elements.studentAssignmentSelect.disabled = false;
    for (const assignment of state.assignments) {
      const option = document.createElement("option");
      option.value = assignment.id;
      option.textContent = assignment.title;
      elements.studentAssignmentSelect.append(option);
    }
    if (state.assignments.some((assignment) => assignment.id === previous)) {
      elements.studentAssignmentSelect.value = previous;
    }
  }

  function renderAssignments() {
    elements.assignmentList.replaceChildren();
    elements.assignmentCount.textContent = String(state.assignments.length);
    if (!state.assignments.length) {
      elements.assignmentList.append(createEmpty("이 강좌에 등록된 과제가 없습니다."));
      return;
    }
    for (const assignment of state.assignments) {
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
      if (isStudent()) {
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

  function renderProjects() {
    elements.projectList.replaceChildren();
    elements.projectCount.textContent = String(state.projects.length);
    if (!state.projects.length) {
      elements.projectList.append(createEmpty("저장된 실험 프로젝트가 없습니다."));
      return;
    }
    for (const project of state.projects) {
      const item = document.createElement("article");
      item.className = "education-list-item project";
      const title = document.createElement("strong");
      title.textContent = project.name;
      const description = document.createElement("p");
      description.textContent =
        `${project.ownerName || "사용자"} · ${project.versionCount} versions`;
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
          project.latestSavedAt ? `saved ${formatDate(project.latestSavedAt)}` : null,
          `updated ${formatDate(project.updatedAt)}`,
        ]),
        actions,
      );
      elements.projectList.append(item);
    }
  }

  function renderSubmissions() {
    elements.submissionList.replaceChildren();
    elements.submissionCount.textContent = String(state.submissions.length);
    if (!state.submissions.length) {
      elements.submissionList.append(createEmpty("제출된 실험이 없습니다."));
      return;
    }
    const assignments = new Map(state.assignments.map((item) => [item.id, item]));
    for (const submission of state.submissions) {
      const assignment = assignments.get(submission.assignmentId);
      const item = document.createElement("article");
      item.className = "education-list-item submission";
      const title = document.createElement("strong");
      title.textContent = `${assignment?.title ?? "과제"} · ${submission.studentName}`;
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
      if (isInstructor()) {
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

  function renderWorkspace() {
    const course = currentCourse();
    elements.professorTools.hidden = !isInstructor();
    elements.studentTools.hidden = !isStudent();
    elements.courseCodePanel.hidden = !course?.joinCode;
    elements.courseJoinCodeDisplay.textContent = course?.joinCode || "-";
    populateStudentAssignmentSelect();
    renderAssignments();
    renderProjects();
    renderSubmissions();
  }

  async function refreshCourseData() {
    const courseId = currentCourseId();
    state.activeProjectId = null;
    if (!courseId) {
      state.assignments = [];
      state.projects = [];
      state.submissions = [];
      renderWorkspace();
      return;
    }
    const [assignments, projects, submissions] = await Promise.all([
      api.listAssignments(courseId),
      api.listProjects(courseId),
      api.listSubmissions(courseId),
    ]);
    state.assignments = assignments;
    state.projects = projects;
    state.submissions = submissions;
    renderWorkspace();
  }

  async function refreshCourses(preferredId = null) {
    state.courses = await api.listCourses();
    populateCourseSelect(preferredId);
    await refreshCourseData();
  }

  async function completeAuthentication(auth, message) {
    state.auth = auth;
    updateIdentity();
    elements.authDialog.close();
    if (!elements.dialog.open) elements.dialog.showModal();
    await refreshCourses();
    setStatus(message, "success");
  }

  async function openWorkspace() {
    await authReady;
    if (!state.auth) {
      if (!elements.authDialog.open) elements.authDialog.showModal();
      return;
    }
    if (!elements.dialog.open) elements.dialog.showModal();
    setStatus("강좌 데이터를 불러오는 중입니다.");
    try {
      await refreshCourses();
      const count = state.courses.length;
      setStatus(`${state.auth.tenant.name}의 강좌 ${count}개를 불러왔습니다.`, "success");
    } catch (error) {
      setStatus(error.message, "error");
    }
  }

  async function saveCurrentProject() {
    const courseId = currentCourseId();
    if (!courseId) throw new Error("강좌를 먼저 선택하세요.");
    const name = elements.studentProjectName.value.trim() || "나의 AI 실험";
    const project = await api.saveProject(courseId, {
      projectId: state.activeProjectId,
      name,
      snapshot: captureExperiment(),
    });
    state.activeProjectId = project.id;
    return project;
  }

  function loadAssignmentStarter(assignmentId) {
    const assignment = state.assignments.find((item) => item.id === assignmentId);
    if (!assignment?.starterSnapshot) throw new Error("이 과제에는 시작 템플릿이 없습니다.");
    const message = restoreExperiment(assignment.starterSnapshot);
    elements.dialog.close();
    setStatus(message || `${assignment.title} 시작 상태를 불러왔습니다.`, "success");
  }

  updateRegistrationFields();
  updateIdentity();
  renderWorkspace();

  authReady = api.me()
    .then((auth) => {
      state.auth = auth;
      updateIdentity();
    })
    .catch((error) => {
      if (error.status !== 401) {
        elements.accountBadge.textContent = "인증 서버 오류";
        console.error(error);
      }
    });

  elements.open.addEventListener("click", openWorkspace);
  elements.authClose.addEventListener("click", () => elements.authDialog.close());
  elements.close.addEventListener("click", () => elements.dialog.close());
  elements.authDialog.addEventListener("click", (event) => {
    if (event.target === elements.authDialog) elements.authDialog.close();
  });
  elements.dialog.addEventListener("click", (event) => {
    if (event.target === elements.dialog) elements.dialog.close();
  });

  elements.registrationMode.addEventListener("change", updateRegistrationFields);

  elements.loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    setAuthStatus("로그인 중입니다.");
    try {
      const auth = await api.login({
        email: elements.loginEmail.value,
        password: elements.loginPassword.value,
      });
      elements.loginPassword.value = "";
      await completeAuthentication(auth, `${auth.user.displayName} 계정으로 로그인했습니다.`);
    } catch (error) {
      setAuthStatus(error.message, "error");
    }
  });

  elements.registerForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const createInstitution = elements.registrationMode.value === "create";
    setAuthStatus("계정과 대학 워크스페이스를 생성하는 중입니다.");
    try {
      const auth = await api.register({
        createInstitution,
        displayName: elements.registerDisplayName.value,
        email: elements.registerEmail.value,
        password: elements.registerPassword.value,
        institutionName: createInstitution ? elements.institutionName.value : undefined,
        institutionSlug: createInstitution ? elements.institutionSlug.value : undefined,
        institutionJoinCode: createInstitution
          ? undefined
          : elements.institutionJoinCode.value,
      });
      elements.registerPassword.value = "";
      await completeAuthentication(
        auth,
        createInstitution
          ? `${auth.tenant.name} 워크스페이스를 생성했습니다.`
          : `${auth.tenant.name}에 학생 계정으로 가입했습니다.`,
      );
    } catch (error) {
      setAuthStatus(error.message, "error");
    }
  });

  elements.logout.addEventListener("click", async () => {
    try {
      await api.logout();
      state.auth = null;
      state.courses = [];
      state.assignments = [];
      state.projects = [];
      state.submissions = [];
      state.activeProjectId = null;
      updateIdentity();
      elements.dialog.close();
      setAuthStatus("로그아웃했습니다.", "success");
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  elements.courseSelect.addEventListener("change", async () => {
    setStatus("강좌 데이터를 불러오는 중입니다.");
    try {
      await refreshCourseData();
      setStatus("강좌 데이터를 갱신했습니다.", "success");
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  elements.courseForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const course = await api.createCourse({
        name: elements.courseName.value,
        code: elements.courseCode.value,
        term: elements.courseTerm.value,
      });
      elements.courseForm.reset();
      await refreshCourses(course.id);
      setStatus(`${course.code} 강좌를 생성했습니다. 가입 코드: ${course.joinCode}`, "success");
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  elements.courseJoinForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const course = await api.joinCourse(elements.courseJoinCode.value);
      elements.courseJoinForm.reset();
      await refreshCourses(course.id);
      setStatus(`${course.code} 강좌에 가입했습니다.`, "success");
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  elements.assignmentForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const courseId = currentCourseId();
    if (!courseId) {
      setStatus("강좌를 먼저 생성하거나 선택하세요.", "error");
      return;
    }
    try {
      const targetPercent = elements.assignmentTargetAccuracy.value;
      const assignment = await api.createAssignment(courseId, {
        title: elements.assignmentTitle.value,
        instructions: elements.assignmentInstructions.value,
        dueAt: elements.assignmentDueAt.value || null,
        requiredFamily: elements.assignmentFamily.value,
        targetAccuracy: targetPercent === "" ? null : Number(targetPercent) / 100,
        starterSnapshot: elements.assignmentIncludeStarter.checked
          ? captureExperiment()
          : null,
      });
      elements.assignmentForm.reset();
      elements.assignmentIncludeStarter.checked = true;
      await refreshCourseData();
      setStatus(`${assignment.title} 과제를 생성했습니다.`, "success");
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  elements.saveProject.addEventListener("click", async () => {
    try {
      const project = await saveCurrentProject();
      await refreshCourseData();
      state.activeProjectId = project.id;
      setStatus(`${project.name}의 ${project.versionCount}번째 버전을 저장했습니다.`, "success");
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

  elements.submit.addEventListener("click", async () => {
    try {
      const assignmentId = elements.studentAssignmentSelect.value;
      if (!assignmentId) throw new Error("제출할 과제를 선택하세요.");
      const project = await saveCurrentProject();
      const snapshot = captureExperiment();
      const submission = await api.submitAssignment(assignmentId, {
        projectId: project.id,
        snapshot,
      });
      await refreshCourseData();
      state.activeProjectId = project.id;
      setStatus(`${submission.attempt}번째 제출을 저장했습니다.`, "success");
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
      } else if (button.dataset.action === "load-starter") {
        loadAssignmentStarter(button.dataset.assignmentId);
      }
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  elements.projectList.addEventListener("click", async (event) => {
    const button = event.target.closest('[data-action="load-project"]');
    if (!button) return;
    try {
      const project = await api.getProject(button.dataset.projectId);
      if (!project.latestSnapshot) throw new Error("저장된 실험 버전이 없습니다.");
      state.activeProjectId = project.id;
      elements.studentProjectName.value = project.name;
      const message = restoreExperiment(project.latestSnapshot);
      elements.dialog.close();
      setStatus(message || `${project.name} 프로젝트를 불러왔습니다.`, "success");
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  elements.submissionList.addEventListener("click", async (event) => {
    const button = event.target.closest('[data-action="grade-submission"]');
    if (!button) return;
    const submissionId = button.dataset.submissionId;
    try {
      const score = elements.submissionList
        .querySelector(`[data-grade-score="${submissionId}"]`).value;
      const feedback = elements.submissionList
        .querySelector(`[data-grade-feedback="${submissionId}"]`).value;
      const submission = await api.gradeSubmission(submissionId, { score, feedback });
      await refreshCourseData();
      setStatus(
        `${submission.studentName} 제출을 ${submission.score}점으로 채점했습니다.`,
        "success",
      );
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  return {
    api,
    open: openWorkspace,
  };
}
