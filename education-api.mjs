export class EducationApiError extends Error {
  constructor(message, { status = 0, code = "request_failed" } = {}) {
    super(message);
    this.name = "EducationApiError";
    this.status = status;
    this.code = code;
  }
}

export class EducationApi {
  constructor(fetcher = globalThis.fetch?.bind(globalThis)) {
    if (!fetcher) throw new Error("Fetch API is required");
    this.fetcher = fetcher;
    this.csrfToken = null;
  }

  async request(path, { method = "GET", body, csrf = true } = {}) {
    const headers = { Accept: "application/json" };
    const request = {
      method,
      credentials: "same-origin",
      headers,
    };
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      request.body = JSON.stringify(body);
    }
    if (csrf && method !== "GET" && this.csrfToken) {
      headers["X-CSRF-Token"] = this.csrfToken;
    }

    let response;
    try {
      response = await this.fetcher(path, request);
    } catch (error) {
      throw new EducationApiError(
        `서버에 연결할 수 없습니다: ${error.message}`,
        { code: "network_error" },
      );
    }

    let data = {};
    const text = await response.text();
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        throw new EducationApiError("서버 응답을 해석할 수 없습니다.", {
          status: response.status,
          code: "invalid_response",
        });
      }
    }
    if (!response.ok) {
      if (response.status === 401) this.csrfToken = null;
      throw new EducationApiError(
        data?.error?.message || `요청이 실패했습니다. (${response.status})`,
        {
          status: response.status,
          code: data?.error?.code || "request_failed",
        },
      );
    }
    if (data.csrfToken) this.csrfToken = data.csrfToken;
    return data;
  }

  async me() {
    return this.request("/api/auth/me");
  }

  async login(payload) {
    return this.request("/api/auth/login", {
      method: "POST",
      body: payload,
      csrf: false,
    });
  }

  async register(payload) {
    return this.request("/api/auth/register", {
      method: "POST",
      body: payload,
      csrf: false,
    });
  }

  async verifyEmail(token) {
    return this.request("/api/auth/verify-email", {
      method: "POST",
      body: { token },
      csrf: false,
    });
  }

  async resendVerification(email) {
    return this.request("/api/auth/resend-verification", {
      method: "POST",
      body: { email },
      csrf: false,
    });
  }

  async requestPasswordReset(email) {
    return this.request("/api/auth/password-reset/request", {
      method: "POST",
      body: { email },
      csrf: false,
    });
  }

  async confirmPasswordReset(token, password) {
    return this.request("/api/auth/password-reset/confirm", {
      method: "POST",
      body: { token, password },
      csrf: false,
    });
  }

  async acceptInvitation(payload) {
    return this.request("/api/auth/invitations/accept", {
      method: "POST",
      body: payload,
      csrf: false,
    });
  }

  async listPublicProviders(tenantSlug) {
    const query = new URLSearchParams({ tenant: tenantSlug });
    return (await this.request(`/api/auth/providers?${query}`)).providers;
  }

  oidcStartUrl(tenantSlug, providerId, returnTo = "/") {
    const query = new URLSearchParams({
      tenant: tenantSlug,
      provider: providerId,
      returnTo,
    });
    return `/api/auth/oidc/start?${query}`;
  }

  async logout() {
    const result = await this.request("/api/auth/logout", {
      method: "POST",
      body: {},
    });
    this.csrfToken = null;
    return result;
  }

  async listCourses() {
    return (await this.request("/api/courses")).courses;
  }

  async createCourse(payload) {
    return (await this.request("/api/courses", {
      method: "POST",
      body: payload,
    })).course;
  }

  async listCourseMembers(courseId) {
    return (await this.request(`/api/courses/${encodeURIComponent(courseId)}/members`))
      .members;
  }

  async removeCourseMember(courseId, userId) {
    return this.request(
      `/api/courses/${encodeURIComponent(courseId)}/members/${encodeURIComponent(userId)}/remove`,
      { method: "POST", body: {} },
    );
  }

  async listInvitations() {
    return (await this.request("/api/admin/invitations")).invitations;
  }

  async createInvitation(payload) {
    return (await this.request("/api/admin/invitations", {
      method: "POST",
      body: payload,
    })).invitation;
  }

  async listAuditEvents(limit = 100) {
    return (await this.request(`/api/admin/audit?limit=${encodeURIComponent(limit)}`))
      .events;
  }

  async listIdentityProviders() {
    return (await this.request("/api/admin/identity-providers")).providers;
  }

  async createIdentityProvider(payload) {
    return (await this.request("/api/admin/identity-providers", {
      method: "POST",
      body: payload,
    })).provider;
  }

  async updateIdentityProvider(providerId, payload) {
    return (await this.request(
      `/api/admin/identity-providers/${encodeURIComponent(providerId)}`,
      { method: "PUT", body: payload },
    )).provider;
  }

  async getLtiCourseService(courseId) {
    return (await this.request(
      `/api/courses/${encodeURIComponent(courseId)}/lti-services`,
    )).service;
  }

  async syncLtiRoster(courseId) {
    return (await this.request(
      `/api/courses/${encodeURIComponent(courseId)}/lti/roster-sync`,
      { method: "POST", body: {} },
    )).sync;
  }

  async sendLtiGrade(submissionId) {
    return (await this.request(
      `/api/submissions/${encodeURIComponent(submissionId)}/lti-grade-passback`,
      { method: "POST", body: {} },
    )).passback;
  }

  async joinCourse(joinCode) {
    return (await this.request("/api/courses/join", {
      method: "POST",
      body: { joinCode },
    })).course;
  }

  async listAssignments(courseId) {
    return (await this.request(`/api/courses/${encodeURIComponent(courseId)}/assignments`))
      .assignments;
  }

  async createAssignment(courseId, payload) {
    return (await this.request(
      `/api/courses/${encodeURIComponent(courseId)}/assignments`,
      { method: "POST", body: payload },
    )).assignment;
  }

  async listProjects(courseId) {
    return (await this.request(`/api/courses/${encodeURIComponent(courseId)}/projects`))
      .projects;
  }

  async saveProject(courseId, payload) {
    return (await this.request(
      `/api/courses/${encodeURIComponent(courseId)}/projects`,
      { method: "POST", body: payload },
    )).project;
  }

  async getProject(projectId) {
    return (await this.request(`/api/projects/${encodeURIComponent(projectId)}`)).project;
  }

  async listSubmissions(courseId) {
    return (await this.request(`/api/courses/${encodeURIComponent(courseId)}/submissions`))
      .submissions;
  }

  async submitAssignment(assignmentId, payload) {
    return (await this.request(
      `/api/assignments/${encodeURIComponent(assignmentId)}/submissions`,
      { method: "POST", body: payload },
    )).submission;
  }

  async gradeSubmission(submissionId, payload) {
    return (await this.request(
      `/api/submissions/${encodeURIComponent(submissionId)}/grade`,
      { method: "POST", body: payload },
    )).submission;
  }
}
