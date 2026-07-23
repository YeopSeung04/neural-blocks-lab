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
    verifyEmailForm: document.getElementById("verifyEmailForm"),
    verifyEmailToken: document.getElementById("verifyEmailToken"),
    resendVerification: document.getElementById("resendVerificationButton"),
    acceptInvitationForm: document.getElementById("acceptInvitationForm"),
    acceptInvitationToken: document.getElementById("acceptInvitationToken"),
    invitationDisplayName: document.getElementById("invitationDisplayName"),
    invitationPassword: document.getElementById("invitationPassword"),
    passwordResetRequestForm: document.getElementById("passwordResetRequestForm"),
    passwordResetEmail: document.getElementById("passwordResetEmail"),
    passwordResetConfirmForm: document.getElementById("passwordResetConfirmForm"),
    passwordResetToken: document.getElementById("passwordResetToken"),
    passwordResetPassword: document.getElementById("passwordResetPassword"),
    ssoLoginForm: document.getElementById("ssoLoginForm"),
    ssoTenantSlug: document.getElementById("ssoTenantSlug"),
    ssoProviderSelect: document.getElementById("ssoProviderSelect"),
    loadSsoProviders: document.getElementById("loadSsoProvidersButton"),
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
    adminTools: document.getElementById("educationAdminTools"),
    studentTools: document.getElementById("educationStudentTools"),
    instructorOperations: document.getElementById("educationInstructorOperations"),
    invitationsPanel: document.getElementById("educationInvitationsPanel"),
    providersPanel: document.getElementById("educationProvidersPanel"),
    ltiServicesPanel: document.getElementById("educationLtiServicesPanel"),
    auditPanel: document.getElementById("educationAuditPanel"),
    courseForm: document.getElementById("courseForm"),
    courseName: document.getElementById("courseName"),
    courseCode: document.getElementById("courseCode"),
    courseTerm: document.getElementById("courseTerm"),
    courseJoinForm: document.getElementById("courseJoinForm"),
    courseJoinCode: document.getElementById("courseJoinCode"),
    invitationForm: document.getElementById("invitationForm"),
    invitationEmail: document.getElementById("invitationEmail"),
    invitationRole: document.getElementById("invitationRole"),
    invitationAttachCourse: document.getElementById("invitationAttachCourse"),
    identityProviderForm: document.getElementById("identityProviderForm"),
    identityProviderId: document.getElementById("identityProviderId"),
    identityProviderKind: document.getElementById("identityProviderKind"),
    identityProviderName: document.getElementById("identityProviderName"),
    identityProviderIssuer: document.getElementById("identityProviderIssuer"),
    identityProviderClientId: document.getElementById("identityProviderClientId"),
    identityProviderDefaultRole: document.getElementById("identityProviderDefaultRole"),
    identityProviderAuthorizationEndpoint: document.getElementById(
      "identityProviderAuthorizationEndpoint",
    ),
    identityProviderTokenEndpoint: document.getElementById("identityProviderTokenEndpoint"),
    identityProviderJwksUri: document.getElementById("identityProviderJwksUri"),
    identityProviderSecretEnv: document.getElementById("identityProviderSecretEnv"),
    identityProviderTokenAuthMethod: document.getElementById(
      "identityProviderTokenAuthMethod",
    ),
    identityProviderPrivateKeyEnv: document.getElementById(
      "identityProviderPrivateKeyEnv",
    ),
    identityProviderPrivateKeyKid: document.getElementById(
      "identityProviderPrivateKeyKid",
    ),
    identityProviderDeploymentId: document.getElementById("identityProviderDeploymentId"),
    identityProviderEnabled: document.getElementById("identityProviderEnabled"),
    identityProviderSubmit: document.getElementById("identityProviderSubmitButton"),
    identityProviderCancel: document.getElementById("identityProviderCancelButton"),
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
    memberList: document.getElementById("educationMembersList"),
    invitationList: document.getElementById("educationInvitationsList"),
    providerList: document.getElementById("educationProvidersList"),
    ltiServices: document.getElementById("educationLtiServices"),
    ltiServiceBadge: document.getElementById("ltiServiceBadge"),
    auditList: document.getElementById("educationAuditList"),
    assignmentCount: document.getElementById("assignmentCount"),
    projectCount: document.getElementById("projectCount"),
    submissionCount: document.getElementById("submissionCount"),
    memberCount: document.getElementById("memberCount"),
    invitationCount: document.getElementById("invitationCount"),
    providerCount: document.getElementById("providerCount"),
    auditCount: document.getElementById("auditCount"),
  };

  const state = {
    auth: null,
    courses: [],
    assignments: [],
    projects: [],
    submissions: [],
    members: [],
    invitations: [],
    providers: [],
    ltiService: null,
    auditEvents: [],
    activeProjectId: null,
  };
  let authReady = Promise.resolve();

  function isInstructor() {
    return ["admin", "professor"].includes(state.auth?.user?.role);
  }

  function isAdmin() {
    return state.auth?.user?.role === "admin";
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
    const verification = user.emailVerified ? "" : " · 이메일 인증 필요";
    elements.accountBadge.textContent =
      `${user.displayName} · ${roleLabel(user.role)}${verification}`;
    elements.userBadge.textContent =
      `${user.displayName} · ${roleLabel(user.role)}${verification}`;
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
        if (submission.status === "graded" && state.ltiService?.ags?.canSendScores) {
          const actions = document.createElement("div");
          actions.className = "education-item-actions";
          const passbackButton = document.createElement("button");
          passbackButton.type = "button";
          passbackButton.dataset.action = "lti-grade-passback";
          passbackButton.dataset.submissionId = submission.id;
          passbackButton.textContent = "LMS로 성적 전송";
          actions.append(passbackButton);
          item.append(actions);
        }
      } else if (submission.feedback) {
        const feedback = document.createElement("p");
        feedback.textContent = `교수 피드백: ${submission.feedback}`;
        item.append(feedback);
      }
      elements.submissionList.append(item);
    }
  }

  function renderMembers() {
    elements.memberList.replaceChildren();
    elements.memberCount.textContent = String(state.members.length);
    if (!state.members.length) {
      elements.memberList.append(createEmpty("현재 강좌에 등록된 사용자가 없습니다."));
      return;
    }
    for (const member of state.members) {
      const item = document.createElement("article");
      item.className = "education-list-item member";
      const title = document.createElement("strong");
      title.textContent = `${member.displayName} · ${roleLabel(member.tenantRole)}`;
      const description = document.createElement("p");
      description.textContent = member.email;
      item.append(
        title,
        description,
        createMeta([
          member.courseRole === "instructor" ? "강좌 교수" : "수강 학생",
          member.emailVerified ? "이메일 인증" : "미인증",
          `${member.submissionCount} submissions`,
          member.authProvider?.toUpperCase(),
        ]),
      );
      if (member.courseRole === "student") {
        const actions = document.createElement("div");
        actions.className = "education-item-actions";
        const removeButton = document.createElement("button");
        removeButton.type = "button";
        removeButton.className = "danger";
        removeButton.dataset.action = "remove-course-member";
        removeButton.dataset.userId = member.id;
        removeButton.textContent = "강좌에서 제외";
        actions.append(removeButton);
        item.append(actions);
      }
      elements.memberList.append(item);
    }
  }

  function renderInvitations() {
    elements.invitationList.replaceChildren();
    elements.invitationCount.textContent = String(state.invitations.length);
    if (!state.invitations.length) {
      elements.invitationList.append(createEmpty("발송한 초대가 없습니다."));
      return;
    }
    for (const invitation of state.invitations) {
      const item = document.createElement("article");
      item.className = "education-list-item invitation";
      const title = document.createElement("strong");
      title.textContent = invitation.email;
      const description = document.createElement("p");
      description.textContent = invitation.acceptedAt
        ? `수락 ${formatDate(invitation.acceptedAt)}`
        : `만료 ${formatDate(invitation.expiresAt)}`;
      item.append(
        title,
        description,
        createMeta([
          roleLabel(invitation.role),
          invitation.courseName,
          invitation.acceptedAt ? "수락 완료" : "대기 중",
        ]),
      );
      elements.invitationList.append(item);
    }
  }

  function renderProviders() {
    elements.providerList.replaceChildren();
    elements.providerCount.textContent = String(state.providers.length);
    if (!state.providers.length) {
      elements.providerList.append(createEmpty("등록된 OIDC/LTI 공급자가 없습니다."));
      return;
    }
    for (const provider of state.providers) {
      const item = document.createElement("article");
      item.className = "education-list-item provider";
      const title = document.createElement("strong");
      title.textContent = provider.name;
      const description = document.createElement("p");
      description.textContent = provider.issuer;
      item.append(
        title,
        description,
        createMeta([
          provider.kind.toUpperCase(),
          provider.enabled ? "활성" : "비활성",
          `기본 ${roleLabel(provider.defaultRole)}`,
          provider.deploymentId ? `deployment ${provider.deploymentId}` : null,
        ]),
      );
      if (provider.kind === "oidc") {
        const link = document.createElement("code");
        link.textContent =
          `${location.origin}/api/auth/oidc/callback`;
        item.append(link);
      } else {
        const link = document.createElement("code");
        link.textContent =
          `${location.origin}/api/auth/lti/login · ${location.origin}/api/auth/lti/launch`;
        item.append(link);
      }
      const actions = document.createElement("div");
      actions.className = "education-item-actions";
      const editButton = document.createElement("button");
      editButton.type = "button";
      editButton.dataset.action = "edit-provider";
      editButton.dataset.providerId = provider.id;
      editButton.textContent = "편집";
      const toggleButton = document.createElement("button");
      toggleButton.type = "button";
      toggleButton.dataset.action = "toggle-provider";
      toggleButton.dataset.providerId = provider.id;
      toggleButton.textContent = provider.enabled ? "비활성화" : "활성화";
      if (provider.enabled) toggleButton.className = "danger";
      actions.append(editButton, toggleButton);
      item.append(actions);
      elements.providerList.append(item);
    }
  }

  function renderLtiServices() {
    elements.ltiServices.replaceChildren();
    const service = state.ltiService;
    elements.ltiServiceBadge.textContent = service?.connected ? "연결됨" : "미연결";
    if (!service?.connected) {
      elements.ltiServices.append(
        createEmpty("이 강좌는 아직 LMS의 LTI Context와 연결되지 않았습니다."),
      );
      return;
    }
    const item = document.createElement("article");
    item.className = "education-list-item lti-service";
    const title = document.createElement("strong");
    title.textContent = `${service.provider.name} · ${service.contextId}`;
    const description = document.createElement("p");
    description.textContent = service.lastRosterSyncAt
      ? `마지막 명단 동기화 ${formatDate(service.lastRosterSyncAt)}`
      : "명단을 아직 동기화하지 않았습니다.";
    item.append(
      title,
      description,
      createMeta([
        service.nrps.available ? "NRPS 사용 가능" : "NRPS 없음",
        service.ags.canCreateLineItems ? "AGS LineItem" : null,
        service.ags.canSendScores ? "AGS Score" : "성적 전송 불가",
        service.provider.enabled ? "공급자 활성" : "공급자 비활성",
      ]),
    );
    if (service.nrps.available && service.provider.enabled) {
      const actions = document.createElement("div");
      actions.className = "education-item-actions";
      const syncButton = document.createElement("button");
      syncButton.type = "button";
      syncButton.className = "primary";
      syncButton.dataset.action = "sync-lti-roster";
      syncButton.textContent = "LMS 명단 동기화";
      actions.append(syncButton);
      item.append(actions);
    }
    elements.ltiServices.append(item);
  }

  function renderAuditEvents() {
    elements.auditList.replaceChildren();
    elements.auditCount.textContent = String(state.auditEvents.length);
    if (!state.auditEvents.length) {
      elements.auditList.append(createEmpty("감사 이벤트가 없습니다."));
      return;
    }
    for (const event of state.auditEvents) {
      const item = document.createElement("article");
      item.className = "education-list-item audit";
      const title = document.createElement("strong");
      title.textContent = event.eventType;
      const description = document.createElement("p");
      description.textContent =
        `${event.userName || "시스템"} · ${formatDate(event.createdAt)}`;
      item.append(
        title,
        description,
        createMeta([
          event.entityType,
          event.ipAddress,
        ]),
      );
      elements.auditList.append(item);
    }
  }

  function renderWorkspace() {
    const course = currentCourse();
    elements.professorTools.hidden = !isInstructor();
    elements.adminTools.hidden = !isAdmin();
    elements.studentTools.hidden = !isStudent();
    elements.instructorOperations.hidden = !isInstructor();
    elements.invitationsPanel.hidden = !isAdmin();
    elements.providersPanel.hidden = !isAdmin();
    elements.ltiServicesPanel.hidden = !isInstructor();
    elements.auditPanel.hidden = !isAdmin();
    elements.courseCodePanel.hidden = !course?.joinCode;
    elements.courseJoinCodeDisplay.textContent = course?.joinCode || "-";
    populateStudentAssignmentSelect();
    renderAssignments();
    renderProjects();
    renderSubmissions();
    renderMembers();
    renderInvitations();
    renderProviders();
    renderLtiServices();
    renderAuditEvents();
  }

  async function refreshCourseData() {
    const courseId = currentCourseId();
    state.activeProjectId = null;
    if (!courseId) {
      state.assignments = [];
      state.projects = [];
      state.submissions = [];
      state.members = [];
      state.ltiService = null;
      renderWorkspace();
      return;
    }
    const requests = [
      api.listAssignments(courseId),
      api.listProjects(courseId),
      api.listSubmissions(courseId),
    ];
    if (isInstructor()) {
      requests.push(api.listCourseMembers(courseId));
      requests.push(api.getLtiCourseService(courseId));
    }
    const [assignments, projects, submissions, members = [], ltiService = null] =
      await Promise.all(requests);
    state.assignments = assignments;
    state.projects = projects;
    state.submissions = submissions;
    state.members = members;
    state.ltiService = ltiService;
    renderWorkspace();
  }

  async function refreshAdminData() {
    if (!isAdmin()) {
      state.invitations = [];
      state.providers = [];
      state.auditEvents = [];
      return;
    }
    const [invitations, providers, auditEvents] = await Promise.all([
      api.listInvitations(),
      api.listIdentityProviders(),
      api.listAuditEvents(100),
    ]);
    state.invitations = invitations;
    state.providers = providers;
    state.auditEvents = auditEvents;
  }

  async function refreshCourses(preferredId = null) {
    state.courses = await api.listCourses();
    populateCourseSelect(preferredId);
    await refreshCourseData();
    await refreshAdminData();
    renderWorkspace();
  }

  async function completeAuthentication(auth, message) {
    state.auth = auth;
    updateIdentity();
    if (!auth.user.emailVerified) {
      if (auth.devVerificationToken) {
        elements.verifyEmailToken.value = auth.devVerificationToken;
      }
      if (!elements.authDialog.open) elements.authDialog.showModal();
      setAuthStatus(
        `${message} 이메일 인증 후 강좌 데이터를 변경할 수 있습니다.`,
        "neutral",
      );
      return;
    }
    if (elements.authDialog.open) elements.authDialog.close();
    if (!elements.dialog.open) elements.dialog.showModal();
    await refreshCourses();
    setStatus(message, "success");
  }

  async function openWorkspace() {
    await authReady;
    if (!state.auth || !state.auth.user.emailVerified) {
      if (!elements.authDialog.open) elements.authDialog.showModal();
      if (state.auth && !state.auth.user.emailVerified) {
        setAuthStatus("이메일 인증을 먼저 완료하세요.", "error");
      }
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

  const startupParameters = new URLSearchParams(location.search);
  const startupVerificationToken = startupParameters.get("verify_email");
  const startupResetToken = startupParameters.get("password_reset");
  const startupInvitationToken = startupParameters.get("invitation");
  if (startupVerificationToken) {
    elements.verifyEmailToken.value = startupVerificationToken;
  }
  if (startupResetToken) {
    elements.passwordResetToken.value = startupResetToken;
  }
  if (startupInvitationToken) {
    elements.acceptInvitationToken.value = startupInvitationToken;
  }
  if (startupVerificationToken || startupResetToken || startupInvitationToken) {
    history.replaceState({}, document.title, location.pathname);
    elements.authDialog.showModal();
  }

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

  function updateIdentityProviderFields() {
    const oidc = elements.identityProviderKind.value === "oidc";
    const privateKeyJwt =
      elements.identityProviderTokenAuthMethod.value === "private_key_jwt";
    elements.identityProviderTokenEndpoint.required = oidc;
    elements.identityProviderDeploymentId.required = !oidc;
    elements.identityProviderPrivateKeyEnv.required = !oidc && privateKeyJwt;
  }

  function resetIdentityProviderForm() {
    elements.identityProviderForm.reset();
    elements.identityProviderId.value = "";
    elements.identityProviderEnabled.checked = true;
    elements.identityProviderSubmit.textContent = "인증 공급자 저장";
    elements.identityProviderCancel.hidden = true;
    elements.identityProviderKind.disabled = false;
    updateIdentityProviderFields();
  }

  function editIdentityProvider(provider) {
    elements.identityProviderId.value = provider.id;
    elements.identityProviderKind.value = provider.kind;
    elements.identityProviderKind.disabled = true;
    elements.identityProviderName.value = provider.name || "";
    elements.identityProviderIssuer.value = provider.issuer || "";
    elements.identityProviderClientId.value = provider.clientId || "";
    elements.identityProviderDefaultRole.value = provider.defaultRole || "student";
    elements.identityProviderAuthorizationEndpoint.value =
      provider.authorizationEndpoint || "";
    elements.identityProviderTokenEndpoint.value = provider.tokenEndpoint || "";
    elements.identityProviderJwksUri.value = provider.jwksUri || "";
    elements.identityProviderSecretEnv.value = provider.clientSecretEnv || "";
    elements.identityProviderTokenAuthMethod.value =
      provider.serviceTokenAuthMethod || "client_secret_basic";
    elements.identityProviderPrivateKeyEnv.value = provider.privateKeyEnv || "";
    elements.identityProviderPrivateKeyKid.value = provider.privateKeyKid || "";
    elements.identityProviderDeploymentId.value = provider.deploymentId || "";
    elements.identityProviderEnabled.checked = provider.enabled;
    elements.identityProviderSubmit.textContent = "인증 공급자 변경 저장";
    elements.identityProviderCancel.hidden = false;
    elements.identityProviderForm.closest("details").open = true;
    updateIdentityProviderFields();
  }

  updateIdentityProviderFields();
  elements.identityProviderKind.addEventListener("change", updateIdentityProviderFields);
  elements.identityProviderTokenAuthMethod.addEventListener(
    "change",
    updateIdentityProviderFields,
  );
  elements.identityProviderCancel.addEventListener("click", resetIdentityProviderForm);

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

  elements.verifyEmailForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await api.verifyEmail(elements.verifyEmailToken.value);
      const auth = await api.me();
      await completeAuthentication(auth, "이메일 인증을 완료했습니다.");
    } catch (error) {
      setAuthStatus(error.message, "error");
    }
  });

  elements.resendVerification.addEventListener("click", async () => {
    const email = state.auth?.user?.email
      || elements.registerEmail.value
      || elements.loginEmail.value;
    if (!email) {
      setAuthStatus("인증 메일을 받을 이메일을 입력하세요.", "error");
      return;
    }
    try {
      const result = await api.resendVerification(email);
      if (result.devVerificationToken) {
        elements.verifyEmailToken.value = result.devVerificationToken;
      }
      setAuthStatus("인증 메일을 다시 발송했습니다.", "success");
    } catch (error) {
      setAuthStatus(error.message, "error");
    }
  });

  elements.acceptInvitationForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const auth = await api.acceptInvitation({
        token: elements.acceptInvitationToken.value,
        displayName: elements.invitationDisplayName.value,
        password: elements.invitationPassword.value,
      });
      elements.invitationPassword.value = "";
      await completeAuthentication(auth, "초대 계정을 생성했습니다.");
    } catch (error) {
      setAuthStatus(error.message, "error");
    }
  });

  elements.passwordResetRequestForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const result = await api.requestPasswordReset(elements.passwordResetEmail.value);
      if (result.devPasswordResetToken) {
        elements.passwordResetToken.value = result.devPasswordResetToken;
      }
      setAuthStatus(
        "계정이 존재하면 비밀번호 재설정 메일이 발송됩니다.",
        "success",
      );
    } catch (error) {
      setAuthStatus(error.message, "error");
    }
  });

  elements.passwordResetConfirmForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await api.confirmPasswordReset(
        elements.passwordResetToken.value,
        elements.passwordResetPassword.value,
      );
      elements.passwordResetPassword.value = "";
      state.auth = null;
      updateIdentity();
      setAuthStatus("비밀번호를 변경했습니다. 새 비밀번호로 로그인하세요.", "success");
    } catch (error) {
      setAuthStatus(error.message, "error");
    }
  });

  elements.loadSsoProviders.addEventListener("click", async () => {
    const tenantSlug = elements.ssoTenantSlug.value.trim();
    if (!tenantSlug) {
      setAuthStatus("워크스페이스 ID를 입력하세요.", "error");
      return;
    }
    try {
      const providers = await api.listPublicProviders(tenantSlug);
      elements.ssoProviderSelect.replaceChildren();
      for (const provider of providers.filter((item) => item.kind === "oidc")) {
        const option = document.createElement("option");
        option.value = provider.id;
        option.textContent = provider.name;
        elements.ssoProviderSelect.append(option);
      }
      elements.ssoProviderSelect.disabled = !elements.ssoProviderSelect.options.length;
      setAuthStatus(
        providers.length
          ? `인증 공급자 ${providers.length}개를 불러왔습니다.`
          : "활성화된 OIDC 공급자가 없습니다.",
        providers.length ? "success" : "error",
      );
    } catch (error) {
      setAuthStatus(error.message, "error");
    }
  });

  elements.ssoLoginForm.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!elements.ssoProviderSelect.value) {
      setAuthStatus("OIDC 공급자를 먼저 조회하고 선택하세요.", "error");
      return;
    }
    location.assign(api.oidcStartUrl(
      elements.ssoTenantSlug.value.trim(),
      elements.ssoProviderSelect.value,
    ));
  });

  elements.logout.addEventListener("click", async () => {
    try {
      await api.logout();
      state.auth = null;
      state.courses = [];
      state.assignments = [];
      state.projects = [];
      state.submissions = [];
      state.members = [];
      state.invitations = [];
      state.providers = [];
      state.ltiService = null;
      state.auditEvents = [];
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

  elements.invitationForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const invitation = await api.createInvitation({
        email: elements.invitationEmail.value,
        role: elements.invitationRole.value,
        courseId: elements.invitationAttachCourse.checked
          ? currentCourseId()
          : null,
      });
      elements.invitationForm.reset();
      elements.invitationAttachCourse.checked = true;
      await refreshAdminData();
      renderWorkspace();
      if (invitation.devInvitationToken) {
        setStatus(
          `${invitation.email} 초대를 발송했습니다. 개발 토큰: ${invitation.devInvitationToken}`,
          "success",
        );
      } else {
        setStatus(`${invitation.email} 초대를 발송했습니다.`, "success");
      }
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  elements.identityProviderForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const providerId = elements.identityProviderId.value;
      const payload = {
        kind: elements.identityProviderKind.value,
        name: elements.identityProviderName.value,
        issuer: elements.identityProviderIssuer.value,
        clientId: elements.identityProviderClientId.value,
        authorizationEndpoint: elements.identityProviderAuthorizationEndpoint.value,
        tokenEndpoint: elements.identityProviderTokenEndpoint.value,
        jwksUri: elements.identityProviderJwksUri.value,
        clientSecretEnv: elements.identityProviderSecretEnv.value,
        serviceTokenAuthMethod: elements.identityProviderTokenAuthMethod.value,
        privateKeyEnv: elements.identityProviderPrivateKeyEnv.value,
        privateKeyKid: elements.identityProviderPrivateKeyKid.value,
        deploymentId: elements.identityProviderDeploymentId.value,
        defaultRole: elements.identityProviderDefaultRole.value,
        enabled: elements.identityProviderEnabled.checked,
      };
      const provider = providerId
        ? await api.updateIdentityProvider(providerId, payload)
        : await api.createIdentityProvider(payload);
      resetIdentityProviderForm();
      await refreshAdminData();
      if (currentCourseId()) await refreshCourseData();
      renderWorkspace();
      setStatus(
        `${provider.name} ${provider.kind.toUpperCase()} 설정을 저장했습니다.`,
        "success",
      );
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  elements.providerList.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-action]");
    if (!button) return;
    const provider = state.providers.find(
      (item) => item.id === button.dataset.providerId,
    );
    if (!provider) return;
    if (button.dataset.action === "edit-provider") {
      editIdentityProvider(provider);
      return;
    }
    if (button.dataset.action !== "toggle-provider") return;
    try {
      await api.updateIdentityProvider(provider.id, {
        enabled: !provider.enabled,
      });
      await refreshAdminData();
      if (currentCourseId()) await refreshCourseData();
      renderWorkspace();
      setStatus(
        `${provider.name} 공급자를 ${provider.enabled ? "비활성화" : "활성화"}했습니다.`,
        "success",
      );
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  elements.ltiServices.addEventListener("click", async (event) => {
    const button = event.target.closest('[data-action="sync-lti-roster"]');
    if (!button) return;
    try {
      button.disabled = true;
      setStatus("LMS에서 강좌 명단을 동기화하는 중입니다.");
      const result = await api.syncLtiRoster(currentCourseId());
      await refreshCourseData();
      setStatus(
        `LMS 명단 ${result.received}명 확인, ${result.enrolled}명 배정, ${result.created}명 계정 생성`,
        "success",
      );
    } catch (error) {
      button.disabled = false;
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
    const button = event.target.closest("[data-action]");
    if (!button) return;
    const submissionId = button.dataset.submissionId;
    if (button.dataset.action === "lti-grade-passback") {
      try {
        button.disabled = true;
        setStatus("채점 결과를 LMS로 전송하는 중입니다.");
        const result = await api.sendLtiGrade(submissionId);
        setStatus(
          `LMS 성적 전송 완료: ${result.score}점 · ${formatDate(result.sentAt)}`,
          "success",
        );
      } catch (error) {
        button.disabled = false;
        setStatus(error.message, "error");
      }
      return;
    }
    if (button.dataset.action !== "grade-submission") return;
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

  elements.memberList.addEventListener("click", async (event) => {
    const button = event.target.closest('[data-action="remove-course-member"]');
    if (!button) return;
    try {
      await api.removeCourseMember(currentCourseId(), button.dataset.userId);
      await refreshCourseData();
      setStatus("학생을 현재 강좌에서 제외했습니다.", "success");
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  return {
    api,
    open: openWorkspace,
  };
}
