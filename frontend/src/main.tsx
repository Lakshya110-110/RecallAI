import { StrictMode, useEffect, useState, useMemo, useRef, type FormEvent } from 'react';
import { createRoot } from 'react-dom/client';
import {
  BookOpen,
  ChartNoAxesCombined,
  ClipboardList,
  Download,
  FileSpreadsheet,
  FileText,
  FileUp,
  House,
  LogOut,
  Menu,
  Moon,
  Plus,
  Send,
  Sun,
  Users,
  X,
  Eye,
  EyeOff,
  RotateCcw,
  AlertCircle,
  AlertTriangle,
  Loader2,
  Search,
  CheckCircle2,
  Clock,
  ChevronDown,
  ChevronUp,
  Star,
  Bell,
  BellRing,
  CheckCheck,
  Upload,
  PenLine,
  type LucideIcon,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import './styles.css';
import type {
  User,
  Role,
  Lesson,
  Submission,
  ConceptResult,
  Theme,
  AppNotification,
  StudentOption,
  ContributorAssignmentItem,
  ContributorUploadHistoryItem,
  LessonContributorsResponse,
  LessonKeywordPoolResponse,
  DashboardAnalyticsResponse,
} from './types';

const API = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';
const SUBJECT = 'Business Forecasting';
const ALLOWED_EMAIL_DOMAIN = '@rvu.edu.in';
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // 10 MB
const ALLOWED_UPLOAD_EXTENSIONS = ['.pdf', '.docx', '.txt'];
const CHART_COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

// ============================================================================
// AUTH & TOKEN HELPERS
// ============================================================================
function isStoredTokenValid(tokenStr: string): boolean {
  if (!tokenStr) return false;
  try {
    const parts = tokenStr.split('.');
    if (parts.length !== 3) return false;
    const payload = JSON.parse(atob(parts[1].replace(/-/g, '+').replace(/_/g, '/')));
    if (typeof payload.exp === 'number' && payload.exp * 1000 < Date.now()) {
      return false;
    }
    return true;
  } catch {
    return false;
  }
}

function loadInitialAuth(): { token: string; user: User | null } {
  try {
    const rawToken = localStorage.getItem('token') || '';
    const rawUser = localStorage.getItem('user');
    const parsedUser = rawUser ? (JSON.parse(rawUser) as User) : null;
    if (!rawToken || !parsedUser || !isStoredTokenValid(rawToken)) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      return { token: '', user: null };
    }
    return { token: rawToken, user: parsedUser };
  } catch {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    return { token: '', user: null };
  }
}

// ============================================================================
// THEME HOOK
// ============================================================================
function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem('recallai-theme') as Theme;
    if (saved === 'aurora-light' || saved === 'neural-dark') return saved;
    return 'aurora-light';
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('recallai-theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(prev => (prev === 'neural-dark' ? 'aurora-light' : 'neural-dark'));
  return [theme, setTheme, toggleTheme] as const;
}

// ============================================================================
// FORMATTERS
// ============================================================================
function formatRelativeTime(dateStr: string): string {
  try {
    const date = new Date(dateStr);
    const now = new Date();
    const diffSec = Math.floor((now.getTime() - date.getTime()) / 1000);
    if (diffSec < 60) return 'Just now';
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHour = Math.floor(diffMin / 60);
    if (diffHour < 24) return `${diffHour}h ago`;
    const diffDays = Math.floor(diffHour / 24);
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch {
    return 'Recently';
  }
}

function formatFileSize(bytes: number | null | undefined): string {
  if (!bytes || bytes <= 0) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatReadinessLabel(status?: string): string {
  switch (status) {
    case 'ready':
    case 'ready_for_assessment':
      return 'Ready for Assessment';
    case 'awaiting_contributors':
      return 'Awaiting Contributors';
    case 'awaiting_uploads':
      return 'Awaiting Uploads';
    case 'processing_documents':
      return 'Processing Documents';
    case 'extraction_failed':
      return 'Extraction Failed';
    default:
      return status || 'Ready';
  }
}

// ============================================================================
// MAIN APP COMPONENT
// ============================================================================
function App() {
  const [theme, , toggleTheme] = useTheme();
  const initialAuth = useMemo(() => loadInitialAuth(), []);
  const [token, setToken] = useState<string>(initialAuth.token);
  const [user, setUser] = useState<User | null>(initialAuth.user);

  const [notice, setNotice] = useState<{ type: 'success' | 'error' | 'info'; message: string } | null>(null);
  const [view, setView] = useState<string>('dashboard');
  const [preselectedLessonId, setPreselectedLessonId] = useState<number | null>(null);
  const [preselectedContributorLessonId, setPreselectedContributorLessonId] = useState<number | null>(null);
  const [mobileNav, setMobileNav] = useState(false);

  const [lessons, setLessons] = useState<Lesson[]>([]);
  const [submissions, setSubmissions] = useState<Submission[]>([]);
  const [contributorAssignments, setContributorAssignments] = useState<ContributorAssignmentItem[]>([]);
  const [contributorUploads, setContributorUploads] = useState<ContributorUploadHistoryItem[]>([]);
  const [studentsList, setStudentsList] = useState<StudentOption[]>([]);
  const [analytics, setAnalytics] = useState<DashboardAnalyticsResponse | null>(null);
  const [notifications, setNotifications] = useState<AppNotification[]>([]);
  const [showNotificationPanel, setShowNotificationPanel] = useState(false);
  const notificationRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(false);
  const [selectedSubmission, setSelectedSubmission] = useState<Submission | null>(null);

  const unreadCount = useMemo(() => notifications.filter(n => !n.is_read).length, [notifications]);

  function clearAuthState(expiredMessage?: string) {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setToken('');
    setUser(null);
    setLessons([]);
    setSubmissions([]);
    setContributorAssignments([]);
    setContributorUploads([]);
    setStudentsList([]);
    setAnalytics(null);
    setNotifications([]);
    setShowNotificationPanel(false);
    if (expiredMessage) {
      setNotice({ type: 'error', message: expiredMessage });
    }
  }

  async function request(path: string, options: RequestInit = {}) {
    const isFormData = typeof FormData !== 'undefined' && options.body instanceof FormData;
    const headers: Record<string, string> = {
      ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...((options.headers as Record<string, string>) || {}),
    };
    const response = await fetch(API + path, { ...options, headers });
    const body = await response.json().catch(() => ({}));
    if (response.status === 401) {
      if (token) {
        clearAuthState('Your session has expired. Please sign in again.');
      }
      throw new Error('Authentication expired');
    }
    if (!response.ok) {
      let msg = `Request failed (${response.status})`;
      if (typeof body.detail === 'string') {
        msg = body.detail;
      } else if (Array.isArray(body.detail) && body.detail.length > 0) {
        const firstErr = body.detail[0];
        const rawMsg = typeof firstErr?.msg === 'string' ? firstErr.msg.replace(/^Value error,\s*/i, '') : '';
        const locParts = Array.isArray(firstErr?.loc)
          ? firstErr.loc.filter((p: any) => p !== 'body' && p !== 'query' && p !== 'path')
          : [];
        const fieldName = locParts.length > 0 ? String(locParts[locParts.length - 1]) : '';
        if (rawMsg.toLowerCase() === 'field required' && fieldName) {
          msg = `Please provide the required ${fieldName.replace(/_/g, ' ')} field.`;
        } else if (rawMsg) {
          msg = rawMsg;
        }
      }
      throw new Error(msg);
    }
    return body;
  }

  async function refresh(silent = false) {
    if (!token) return;
    if (!silent) setLoading(true);
    try {
      const [meData, lessonData, submissionData, notifData, analyticsData] = await Promise.all([
        request('/auth/me').catch(err => {
          if (err.message === 'Authentication expired') throw err;
          return null;
        }),
        request('/lessons'),
        request('/submissions'),
        request('/notifications').catch(() => []),
        request('/analytics/dashboard').catch(() => null),
      ]);
      if (meData) {
        const mergedUser: User = {
          ...meData,
          login_portal: user?.login_portal || meData.login_portal || meData.role,
        };
        setUser(mergedUser);
        localStorage.setItem('user', JSON.stringify(mergedUser));
      }
      setLessons(lessonData);
      setSubmissions(submissionData);
      if (Array.isArray(notifData)) setNotifications(notifData);
      if (analyticsData) setAnalytics(analyticsData);

      const portalRole: Role =
        user?.login_portal || meData?.login_portal || meData?.role || user?.role || 'student';

      if (portalRole === 'contributor' || meData?.role === 'contributor') {
        const [contribData, uploadHistoryData, eligibleRows] = await Promise.all([
          request('/contributor/assignments').catch(() => []),
          request('/contributor/uploads').catch(() => []),
          request('/contributors/eligible').catch(() => []),
        ]);
        if (Array.isArray(contribData)) setContributorAssignments(contribData);
        if (Array.isArray(uploadHistoryData)) setContributorUploads(uploadHistoryData);
        if (Array.isArray(eligibleRows)) setStudentsList(eligibleRows);
      } else if (portalRole === 'faculty') {
        const studentRows = await request('/contributors/eligible').catch(() => []);
        if (Array.isArray(studentRows)) setStudentsList(studentRows);
      }
    } catch (err: any) {
      if (!silent && err.message !== 'Authentication expired') {
        setNotice({ type: 'error', message: err.message || 'Failed to load data' });
      }
    } finally {
      if (!silent) setLoading(false);
    }
  }

  async function handleMarkNotificationRead(id: number) {
    setNotifications(prev => prev.map(n => (n.id === id ? { ...n, is_read: true } : n)));
    try {
      await request(`/notifications/${id}/read`, { method: 'POST' });
    } catch {
      // ignore
    }
  }

  async function handleMarkAllAsRead() {
    setNotifications(prev => prev.map(n => ({ ...n, is_read: true })));
    try {
      await request('/notifications/read-all', { method: 'POST' });
    } catch {
      // ignore
    }
  }

  function handleNotificationClick(notif: AppNotification) {
    if (!notif.is_read) {
      handleMarkNotificationRead(notif.id);
    }
    setShowNotificationPanel(false);
    const activePortal: Role =
      user?.role === 'faculty'
        ? 'faculty'
        : user?.role === 'contributor' || user?.login_portal === 'contributor'
          ? 'contributor'
          : 'student';

    if (activePortal === 'contributor') {
      if (
        notif.category === 'submission' ||
        notif.category === 'assessment_complete' ||
        notif.category === 'assessment_failed' ||
        notif.link_view === 'submissions' ||
        notif.link_view === 'results'
      ) {
        setView('my-submissions');
        if (notif.link_id) {
          const sub = submissions.find(s => s.id === notif.link_id);
          if (sub) setSelectedSubmission(sub);
        }
        return;
      }
      if (notif.link_id) setPreselectedContributorLessonId(notif.link_id);
      setView('submit-document');
      return;
    }

    if (notif.link_view) {
      setView(notif.link_view);
    }
    if (notif.link_id) {
      const sub = submissions.find(s => s.id === notif.link_id);
      if (sub) {
        setSelectedSubmission(sub);
      }
    }
  }

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (notificationRef.current && !notificationRef.current.contains(event.target as Node)) {
        setShowNotificationPanel(false);
      }
    }
    if (showNotificationPanel) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showNotificationPanel]);

  useEffect(() => {
    if (token) {
      refresh();
      const hasProcessingSub = submissions.some(
        s => s.assessment_status === 'processing' || s.assessment_status === 'pending'
      );
      const hasProcessingDoc =
        contributorAssignments.some(
          c => c.processing_status === 'processing' || c.processing_status === 'uploaded'
        ) || lessons.some(l => l.readiness_status === 'processing_documents');

      const pollDelay = hasProcessingSub || hasProcessingDoc ? 3000 : 8000;
      const interval = setInterval(() => {
        refresh(true);
      }, pollDelay);
      return () => clearInterval(interval);
    }
  }, [token, view]);

  useEffect(() => {
    if (selectedSubmission) {
      const updated = submissions.find(s => s.id === selectedSubmission.id);
      if (updated) setSelectedSubmission(updated);
    }
  }, [submissions]);

  function handleSignedIn(nextToken: string, nextUser: User) {
    setToken(nextToken);
    setUser(nextUser);
    localStorage.setItem('token', nextToken);
    localStorage.setItem('user', JSON.stringify(nextUser));
    setView('dashboard');
    setNotice(null);
  }

  async function handleSignOut() {
    try {
      if (token) await request('/auth/logout', { method: 'POST' });
    } catch {
      // ignore
    }
    clearAuthState();
    setNotice(null);
    setView('dashboard');
  }

  async function handleDownload(path: string, filename: string) {
    try {
      const response = await fetch(API + path, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) {
        throw new Error('Report is not ready or download failed.');
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setNotice({ type: 'success', message: `Downloaded ${filename}` });
    } catch (err: any) {
      setNotice({ type: 'error', message: err.message || 'Download failed' });
    }
  }

  if (!user || !token) {
    return (
      <AuthScreen
        theme={theme}
        toggleTheme={toggleTheme}
        onSignedIn={handleSignedIn}
        notice={notice}
        setNotice={setNotice}
      />
    );
  }

  const activePortal: Role =
    user.role === 'faculty'
      ? 'faculty'
      : user.role === 'contributor' || user.login_portal === 'contributor'
        ? 'contributor'
        : 'student';

  const isFaculty = activePortal === 'faculty';
  const isContributorPortal = activePortal === 'contributor';
  const isStudentPortal = activePortal === 'student';

  const navItems: { key: string; label: string; icon: LucideIcon; count?: number }[] = isFaculty
    ? [
        { key: 'dashboard', label: 'Overview', icon: House },
        { key: 'lessons', label: 'Manage Sessions', icon: BookOpen, count: lessons.length },
        { key: 'submissions', label: 'Submissions', icon: Users, count: submissions.length },
        { key: 'reports', label: 'Reports', icon: FileSpreadsheet },
      ]
    : isContributorPortal
      ? [
          {
            key: 'dashboard',
            label: 'Overview',
            icon: House,
            count: contributorAssignments.length,
          },
          { key: 'submit-document', label: 'Submit Reference Report', icon: FileUp },
          {
            key: 'manage-sessions',
            label: 'Manage Sessions',
            icon: BookOpen,
            count: lessons.filter(l => l.is_creator).length || undefined,
          },
          { key: 'write-reflection', label: 'Write Reflection', icon: Send },
          {
            key: 'my-submissions',
            label: 'My Submissions',
            icon: ClipboardList,
            count: contributorUploads.length + submissions.length,
          },
        ]
      : [
          { key: 'dashboard', label: 'Overview', icon: House },
          { key: 'lessons', label: 'Available Sessions', icon: BookOpen, count: lessons.length },
          { key: 'submit', label: 'Submit Reflection', icon: Send },
          { key: 'submissions', label: 'My Submissions', icon: ClipboardList, count: submissions.length },
          {
            key: 'results',
            label: 'Assessment Results',
            icon: ChartNoAxesCombined,
            count: submissions.filter(s => s.assessment_status === 'completed').length,
          },
        ];

  const currentNavLabel = navItems.find(n => n.key === view)?.label || 'Overview';

  return (
    <div className="app-shell">
      {/* COMPACT SIDEBAR */}
      <aside className={`sidebar ${mobileNav ? 'mobile-open' : ''}`}>
        <div className="brand-section">
          <div className="brand-container">
            <div className="brand-icon">
              <ChartNoAxesCombined size={16} />
            </div>
            <span className="brand-name">RecallAI</span>
          </div>
          {mobileNav && (
            <button className="btn-icon" onClick={() => setMobileNav(false)} aria-label="Close navigation">
              <X size={16} />
            </button>
          )}
        </div>

        <nav className="nav-group" aria-label="Main Navigation">
          {navItems.map(item => {
            const Icon = item.icon;
            const active = view === item.key;
            return (
              <button
                key={item.key}
                className={`nav-item ${active ? 'active' : ''}`}
                onClick={() => {
                  setView(item.key);
                  setMobileNav(false);
                }}
              >
                <Icon size={16} />
                <span>{item.label}</span>
                {typeof item.count === 'number' && item.count > 0 && (
                  <span className="nav-item-badge">{item.count}</span>
                )}
              </button>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-user-card">
            <div className="user-avatar">{user.name.slice(0, 1).toUpperCase()}</div>
            <div className="user-details">
              <span className="user-name">{user.name}</span>
              <span className="user-role-badge">{activePortal}</span>
            </div>
          </div>

          <button className="sidebar-logout-btn" onClick={handleSignOut}>
            <LogOut size={14} />
            <span>Log out</span>
          </button>
        </div>
      </aside>

      {/* MAIN WORKSPACE */}
      <div className="main-wrapper">
        {/* TOPBAR */}
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="mobile-menu-trigger"
              onClick={() => setMobileNav(true)}
              aria-label="Open navigation"
            >
              <Menu size={18} />
            </button>
            <span className="topbar-page-title">{currentNavLabel}</span>
          </div>

          <div className="topbar-right">
            {/* NOTIFICATIONS */}
            <div className="notification-wrapper" ref={notificationRef}>
              <button
                className="btn-icon"
                onClick={() => setShowNotificationPanel(prev => !prev)}
                title="Notifications"
                aria-label={`Notifications, ${unreadCount} unread`}
              >
                {unreadCount > 0 ? <BellRing size={16} /> : <Bell size={16} />}
                {unreadCount > 0 && (
                  <span className="notification-badge">{unreadCount > 9 ? '9+' : unreadCount}</span>
                )}
              </button>

              {showNotificationPanel && (
                <div className="notification-panel">
                  <div className="notification-panel-header">
                    <span className="notification-title">Notifications</span>
                    {unreadCount > 0 && (
                      <button className="btn btn-ghost btn-sm" onClick={handleMarkAllAsRead}>
                        <CheckCheck size={13} />
                        <span>Mark all read</span>
                      </button>
                    )}
                  </div>

                  <div className="notification-list">
                    {notifications.length === 0 ? (
                      <div className="empty-state" style={{ padding: '24px 16px' }}>
                        <span className="empty-state-description">No notifications yet.</span>
                      </div>
                    ) : (
                      notifications.map(n => (
                        <div
                          key={n.id}
                          className={`notification-item ${!n.is_read ? 'unread' : ''}`}
                          onClick={() => handleNotificationClick(n)}
                        >
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
                              <span className="notification-item-title">{n.title}</span>
                              <span className="notification-item-time">{formatRelativeTime(n.created_at)}</span>
                            </div>
                            <p className="notification-item-msg">{n.message}</p>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* THEME TOGGLE */}
            <button
              className="btn-icon"
              onClick={toggleTheme}
              title={`Switch to ${theme === 'neural-dark' ? 'Light' : 'Dark'} mode`}
              aria-label="Toggle theme"
            >
              {theme === 'neural-dark' ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </header>

        {/* NOTICE ALERT */}
        {notice && (
          <div style={{ maxWidth: 1160, width: '100%', margin: '16px auto 0', padding: '0 24px' }}>
            <div className={notice.type === 'error' ? 'form-error-alert' : 'form-success-alert'} role="alert">
              {notice.type === 'error' ? <AlertCircle size={15} /> : <CheckCircle2 size={15} />}
              <span>{notice.message}</span>
              <button
                style={{
                  marginLeft: 'auto',
                  background: 'transparent',
                  border: 'none',
                  color: 'inherit',
                  cursor: 'pointer',
                }}
                onClick={() => setNotice(null)}
                aria-label="Dismiss alert"
              >
                <X size={14} />
              </button>
            </div>
          </div>
        )}

        {/* MAIN PAGE CONTENT */}
        <main className="page-container">
          {isContributorPortal && (
            <ContributorPortalWorkspace
              currentUser={user}
              view={view}
              setView={setView}
              assignments={contributorAssignments}
              uploads={contributorUploads}
              lessons={lessons}
              submissions={submissions}
              students={studentsList}
              analytics={analytics}
              loading={loading}
              preselectedLessonId={preselectedContributorLessonId}
              setPreselectedLessonId={setPreselectedContributorLessonId}
              preselectedReflectionLessonId={preselectedLessonId}
              setPreselectedReflectionLessonId={setPreselectedLessonId}
              onSelectSubmission={setSelectedSubmission}
              onDownload={handleDownload}
              request={request}
              onRefresh={() => refresh()}
              onSuccess={msg => setNotice({ type: 'success', message: msg })}
              onError={msg => setNotice({ type: 'error', message: msg })}
            />
          )}

          {isStudentPortal && (
            <>
              {view === 'dashboard' && (
                <StudentDashboardOverview
                  lessons={lessons}
                  submissions={submissions}
                  analytics={analytics}
                  loading={loading}
                  onStartReflection={lessonId => {
                    setPreselectedLessonId(lessonId);
                    setView('submit');
                  }}
                  onSelectSubmission={setSelectedSubmission}
                  onNavigate={setView}
                />
              )}

              {view === 'lessons' && (
                <StudentLessonsView
                  lessons={lessons}
                  submissions={submissions}
                  loading={loading}
                  onStartReflection={lessonId => {
                    setPreselectedLessonId(lessonId);
                    setView('submit');
                  }}
                  onViewSubmission={setSelectedSubmission}
                />
              )}

              {view === 'submit' && (
                <ReflectionSubmitView
                  lessons={lessons}
                  submissions={submissions}
                  initialLessonId={preselectedLessonId}
                  request={request}
                  onDone={() => {
                    refresh();
                    setView('submissions');
                    setNotice({ type: 'success', message: 'Reflection submitted for assessment.' });
                  }}
                  onError={msg => setNotice({ type: 'error', message: msg })}
                />
              )}

              {view === 'submissions' && (
                <SubmissionsListView
                  faculty={false}
                  submissions={submissions}
                  loading={loading}
                  request={request}
                  onRefresh={() => refresh()}
                  onSelectSubmission={setSelectedSubmission}
                  onDownload={handleDownload}
                  onError={msg => setNotice({ type: 'error', message: msg })}
                />
              )}

              {view === 'results' && (
                <MyResultsView
                  submissions={submissions.filter(s => s.assessment_status === 'completed')}
                  onSelectSubmission={setSelectedSubmission}
                  onDownload={handleDownload}
                />
              )}
            </>
          )}

          {isFaculty && (
            <>
              {view === 'dashboard' && (
                <FacultyDashboardOverview
                  lessons={lessons}
                  submissions={submissions}
                  analytics={analytics}
                  loading={loading}
                  onNavigate={setView}
                  onSelectSubmission={setSelectedSubmission}
                />
              )}

              {view === 'lessons' && (
                <FacultyLessonsView
                  currentUser={user}
                  lessons={lessons}
                  submissions={submissions}
                  students={studentsList}
                  request={request}
                  onRefresh={() => refresh()}
                  onSuccess={msg => setNotice({ type: 'success', message: msg })}
                  onError={msg => setNotice({ type: 'error', message: msg })}
                />
              )}

              {view === 'submissions' && (
                <SubmissionsListView
                  faculty={true}
                  submissions={submissions}
                  loading={loading}
                  request={request}
                  onRefresh={() => refresh()}
                  onSelectSubmission={setSelectedSubmission}
                  onDownload={handleDownload}
                  onError={msg => setNotice({ type: 'error', message: msg })}
                />
              )}

              {view === 'reports' && (
                <ReportsView onDownload={handleDownload} submissionsCount={submissions.length} />
              )}
            </>
          )}
        </main>
      </div>

      {/* SUBMISSION DETAILS DRAWER */}
      {selectedSubmission && (
        <SubmissionDetailDrawer
          submission={selectedSubmission}
          faculty={isFaculty}
          onClose={() => setSelectedSubmission(null)}
          onDownload={handleDownload}
          request={request}
          onRefresh={() => refresh()}
          onError={msg => setNotice({ type: 'error', message: msg })}
        />
      )}
    </div>
  );
}

// ============================================================================
// 1 & 2. COMPACT AUTHENTICATION SCREEN (LOGIN & CREATE ACCOUNT)
// ============================================================================
const ACCOUNT_TYPES: { role: Role; label: string; loginLabel: string }[] = [
  { role: 'faculty', label: 'Faculty', loginLabel: 'Faculty Login' },
  { role: 'student', label: 'Student', loginLabel: 'Student Login' },
  { role: 'contributor', label: 'Contributor', loginLabel: 'Contributor Login' },
];

function AuthScreen({
  theme,
  toggleTheme,
  onSignedIn,
  notice,
  setNotice,
}: {
  theme: Theme;
  toggleTheme: () => void;
  onSignedIn: (token: string, user: User) => void;
  notice: { type: string; message: string } | null;
  setNotice: (n: any) => void;
}) {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [role, setRole] = useState<Role>('faculty');
  const [showPassword, setShowPassword] = useState(false);
  const [form, setForm] = useState({
    name: '',
    email: '',
    password: '',
    signup_code: '',
  });
  const [loading, setLoading] = useState(false);

  function handleSelectRole(nextRole: Role) {
    setRole(nextRole);
    setNotice(null);
  }

  function handleSwitchMode(nextMode: 'login' | 'register') {
    setMode(nextMode);
    setNotice(null);
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setNotice(null);
    const normalizedEmail = form.email.trim().toLowerCase();
    if (!normalizedEmail.endsWith(ALLOWED_EMAIL_DOMAIN) || normalizedEmail.split('@')[0].length === 0) {
      setNotice({
        type: 'error',
        message: `Email must be a valid college email address ending in ${ALLOWED_EMAIL_DOMAIN}`,
      });
      return;
    }

    setLoading(true);
    try {
      if (mode === 'login') {
        const res = await fetch(API + '/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            email: normalizedEmail,
            password: form.password,
            expected_role: role,
          }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          const msg =
            typeof data.detail === 'string'
              ? data.detail
              : data.detail?.[0]?.msg || 'Invalid credentials. Please try again.';
          throw new Error(msg);
        }
        onSignedIn(data.token, { ...data.user, login_portal: role });
      } else {
        if (!form.name.trim() || form.name.trim().length < 2) {
          throw new Error('Please enter your name (at least 2 characters).');
        }
        if (form.password.length < 8) {
          throw new Error('Password must be at least 8 characters.');
        }
        if ((role === 'faculty' || role === 'contributor') && !form.signup_code.trim()) {
          throw new Error('A valid faculty signup code is required for Faculty and Contributor registration.');
        }
        const res = await fetch(API + '/auth/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: form.name.trim(),
            email: normalizedEmail,
            password: form.password,
            confirm_password: form.password,
            role,
            signup_code: role === 'faculty' || role === 'contributor' ? form.signup_code.trim() : undefined,
            invitation_code: role === 'faculty' || role === 'contributor' ? form.signup_code.trim() : undefined,
          }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          const msg =
            typeof data.detail === 'string'
              ? data.detail
              : data.detail?.[0]?.msg || 'Registration failed.';
          throw new Error(msg);
        }
        setMode('login');
        setForm(f => ({ ...f, password: '', signup_code: '' }));
        setNotice({
          type: 'success',
          message: 'Account created successfully. Please sign in.',
        });
      }
    } catch (err: any) {
      setNotice({ type: 'error', message: err.message || 'Authentication failed' });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-container">
      <div className="auth-theme-toggle">
        <button
          className="btn-icon"
          onClick={toggleTheme}
          title={`Switch to ${theme === 'neural-dark' ? 'Light' : 'Dark'} mode`}
          aria-label="Toggle theme"
        >
          {theme === 'neural-dark' ? <Sun size={16} /> : <Moon size={16} />}
        </button>
      </div>

      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-brand-row">
            <div className="auth-brand-logo">
              <ChartNoAxesCombined size={16} />
            </div>
            <span className="auth-brand-name">RecallAI</span>
          </div>
          <h1 className="auth-title">{mode === 'login' ? 'Welcome to RecallAI' : 'Create Account'}</h1>
          <p className="auth-subtitle">Your Business Forecasting assessment platform.</p>
        </div>

        {/* THREE COMPACT ACCOUNT-TYPE SELECTORS */}
        <div className="auth-portal-tabs" role="tablist" aria-label="Account Type">
          {ACCOUNT_TYPES.map(item => {
            const active = role === item.role;
            return (
              <button
                key={item.role}
                type="button"
                role="tab"
                aria-selected={active}
                className={`auth-portal-tab ${active ? 'active' : ''}`}
                onClick={() => handleSelectRole(item.role)}
              >
                {mode === 'login' ? item.loginLabel : item.label}
              </button>
            );
          })}
        </div>

        {/* COMPACT ACCESSIBLE ALERT DIRECTLY ABOVE FORM */}
        {notice && (
          <div className={notice.type === 'error' ? 'form-error-alert' : 'form-success-alert'} role="alert">
            {notice.type === 'error' ? <AlertCircle size={15} /> : <CheckCircle2 size={15} />}
            <span>{notice.message}</span>
          </div>
        )}

        {/* SINGLE COMPACT FORM */}
        <form className="auth-form" onSubmit={handleSubmit}>
          {mode === 'register' && (
            <div className="form-group">
              <label className="form-label" htmlFor="auth-name">
                Name
              </label>
              <input
                id="auth-name"
                type="text"
                required
                placeholder="Full name"
                value={form.name}
                onChange={e => {
                  setForm({ ...form, name: e.target.value });
                  if (notice?.type === 'error') setNotice(null);
                }}
              />
            </div>
          )}

          <div className="form-group">
            <label className="form-label" htmlFor="auth-email">
              College Email ({ALLOWED_EMAIL_DOMAIN})
            </label>
            <input
              id="auth-email"
              type="email"
              required
              placeholder={`name${ALLOWED_EMAIL_DOMAIN}`}
              value={form.email}
              onChange={e => {
                setForm({ ...form, email: e.target.value });
                if (notice?.type === 'error') setNotice(null);
              }}
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="auth-password">
              Password
            </label>
            <div className="password-input-wrapper">
              <input
                id="auth-password"
                type={showPassword ? 'text' : 'password'}
                required
                minLength={8}
                placeholder="Enter password"
                value={form.password}
                onChange={e => {
                  setForm({ ...form, password: e.target.value });
                  if (notice?.type === 'error') setNotice(null);
                }}
              />
              <button
                type="button"
                className="password-toggle-btn"
                onClick={() => setShowPassword(prev => !prev)}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? <EyeOff size={15} /> : <Eye size={15} />}
              </button>
            </div>
          </div>

          {mode === 'register' && (role === 'faculty' || role === 'contributor') && (
            <div className="form-group">
              <label className="form-label" htmlFor="auth-invite-code">
                Faculty / Contributor Signup Code
              </label>
              <input
                id="auth-invite-code"
                type="password"
                required
                placeholder="Enter faculty signup code"
                value={form.signup_code}
                onChange={e => {
                  setForm({ ...form, signup_code: e.target.value });
                  if (notice?.type === 'error') setNotice(null);
                }}
              />
            </div>
          )}

          <button type="submit" className="btn btn-primary btn-lg" disabled={loading} style={{ width: '100%' }}>
            {loading ? (
              <>
                <Loader2 size={15} className="spin" />
                <span>{mode === 'login' ? 'Signing in...' : 'Creating account...'}</span>
              </>
            ) : (
              <span>{mode === 'login' ? 'Sign In' : 'Create Account'}</span>
            )}
          </button>
        </form>

        <div className="auth-footer-links">
          {mode === 'login' ? (
            <div>
              Don't have an account?{' '}
              <button type="button" className="auth-link-btn" onClick={() => handleSwitchMode('register')}>
                Create Account
              </button>
            </div>
          ) : (
            <div>
              Already have an account?{' '}
              <button type="button" className="auth-link-btn" onClick={() => handleSwitchMode('login')}>
                Back to Login
              </button>
            </div>
          )}

          <div className="auth-role-switch-hint">
            <span>Signing in as {ACCOUNT_TYPES.find(a => a.role === role)?.label}.</span>
            <span>Switch:</span>
            {ACCOUNT_TYPES.filter(a => a.role !== role).map(a => (
              <button
                key={a.role}
                type="button"
                className="auth-link-btn"
                style={{ fontSize: '12px' }}
                onClick={() => handleSelectRole(a.role)}
              >
                {a.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// 4. FACULTY DASHBOARD & MANAGE SESSIONS (CLEAN & UNCLUTTERED WITH REAL CHARTS)
// ============================================================================
function FacultyDashboardOverview({
  lessons,
  submissions,
  analytics,
  loading,
  onNavigate,
  onSelectSubmission,
}: {
  lessons: Lesson[];
  submissions: Submission[];
  analytics: DashboardAnalyticsResponse | null;
  loading: boolean;
  onNavigate: (view: string) => void;
  onSelectSubmission: (sub: Submission) => void;
}) {
  const completed = submissions.filter(s => s.assessment_status === 'completed');
  const pending = submissions.filter(
    s => s.assessment_status === 'pending' || s.assessment_status === 'processing' || s.assessment_status === 'received'
  );
  const avgRating = completed.length
    ? (completed.reduce((acc, curr) => acc + (curr.rating || 0), 0) / completed.length).toFixed(1)
    : null;

  const ratingDist = analytics?.rating_distribution || [];
  const completionData = (analytics?.submission_completion || []).filter(item => item.value > 0);
  const sessionComp = analytics?.session_comparison || [];
  const conceptCov = analytics?.concept_coverage || [];

  return (
    <>
      <div className="page-header">
        <div className="page-header-text">
          <h1 className="page-title">Faculty Overview</h1>
          <p className="page-description">
            Monitor sessions, contributor reference-report readiness, student reflections, and assessment analytics.
          </p>
        </div>
        <button className="btn btn-primary" onClick={() => onNavigate('lessons')}>
          <Plus size={15} />
          <span>Manage Sessions</span>
        </button>
      </div>

      <div className="kpi-grid">
        <div className="kpi-card">
          <span className="kpi-label">Published Sessions</span>
          <span className="kpi-value">{loading ? '—' : lessons.length}</span>
          <span className="kpi-meta">{analytics?.ready_sessions ?? 0} ready for assessment</span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Total Submissions</span>
          <span className="kpi-value">{loading ? '—' : submissions.length}</span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Evaluated</span>
          <span className="kpi-value">{loading ? '—' : completed.length}</span>
          {pending.length > 0 && <span className="kpi-meta">{pending.length} processing</span>}
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Average C4 Rating</span>
          <span className="kpi-value">{avgRating !== null ? `${avgRating} / 5` : '—'}</span>
        </div>
      </div>

      {/* REAL DATABASE-DRIVEN ANALYTICS CHARTS */}
      <div className="analytics-charts-grid">
        {/* 1. Rating Distribution (1 to 5) */}
        <div className="chart-card">
          <div className="chart-card-header">
            <span className="chart-card-title">Rating Distribution (1–5)</span>
            <span className="chart-card-subtitle">Completed student assessments</span>
          </div>
          {completed.length === 0 ? (
            <div className="empty-state" style={{ padding: '40px 16px' }}>
              <span className="empty-state-description">No evaluated reflections yet.</span>
            </div>
          ) : (
            <div className="chart-container-box">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={ratingDist} margin={{ top: 8, right: 12, left: -16, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="rating" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                  <Tooltip />
                  <Bar dataKey="count" name="Students" fill="#6366f1" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* 2. Submission Completion */}
        <div className="chart-card">
          <div className="chart-card-header">
            <span className="chart-card-title">Submission Completion</span>
            <span className="chart-card-subtitle">Evaluation pipeline breakdown</span>
          </div>
          {completionData.length === 0 ? (
            <div className="empty-state" style={{ padding: '40px 16px' }}>
              <span className="empty-state-description">No submission activity recorded yet.</span>
            </div>
          ) : (
            <div className="chart-container-box">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={completionData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={45}
                    outerRadius={72}
                    paddingAngle={3}
                  >
                    {completionData.map((_, idx) => (
                      <Cell key={idx} fill={CHART_COLORS[idx % CHART_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: '12px' }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* 3. Session Comparison & Performance Trend */}
        <div className="chart-card">
          <div className="chart-card-header">
            <span className="chart-card-title">Session Performance Comparison</span>
            <span className="chart-card-subtitle">Average rating & submissions by session</span>
          </div>
          {sessionComp.length === 0 ? (
            <div className="empty-state" style={{ padding: '40px 16px' }}>
              <span className="empty-state-description">Publish a session to view comparison metrics.</span>
            </div>
          ) : (
            <div className="chart-container-box">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={sessionComp} margin={{ top: 8, right: 12, left: -16, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="session" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                  <YAxis domain={[0, 5]} tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: '12px' }} />
                  <Bar dataKey="average_rating" name="Avg Rating (/5)" fill="#6366f1" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="submissions" name="Submissions" fill="#10b981" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* 4. Concept Coverage Across Sessions */}
        <div className="chart-card">
          <div className="chart-card-header">
            <span className="chart-card-title">Concept Coverage Trend</span>
            <span className="chart-card-subtitle">Avg matched concepts out of session pool</span>
          </div>
          {conceptCov.length === 0 ? (
            <div className="empty-state" style={{ padding: '40px 16px' }}>
              <span className="empty-state-description">Concept coverage data will appear after evaluations.</span>
            </div>
          ) : (
            <div className="chart-container-box">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={conceptCov} margin={{ top: 8, right: 12, left: -16, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="session" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                  <YAxis domain={[0, 15]} tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: '12px' }} />
                  <Line
                    type="monotone"
                    dataKey="avg_matched_concepts"
                    name="Avg Matched Concepts"
                    stroke="#6366f1"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      {/* SESSION OVERVIEW TABLE */}
      <div className="table-card" style={{ marginBottom: '20px' }}>
        <div className="table-card-header">
          <span className="table-card-title">Session Overview Table</span>
          <button className="btn btn-secondary btn-sm" onClick={() => onNavigate('lessons')}>
            Manage All Sessions
          </button>
        </div>

        {lessons.length === 0 ? (
          <div className="empty-state">
            <span className="empty-state-title">No sessions created yet</span>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Session Title</th>
                  <th>Created By</th>
                  <th>Assigned Contributors</th>
                  <th>Report Processing</th>
                  <th>Student Submissions</th>
                  <th>Assessment Readiness</th>
                </tr>
              </thead>
              <tbody>
                {lessons.map(l => (
                  <tr key={l.id}>
                    <td>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        <span className="cell-lesson-title">{l.title}</span>
                        <span style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>
                          {l.session_date || SUBJECT}
                        </span>
                      </div>
                    </td>
                    <td>
                      <span style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>
                        {l.creator_name || 'Faculty'} ({l.creator_role || 'faculty'})
                      </span>
                    </td>
                    <td>
                      <span className="cell-mono">
                        {l.assessment_mode === 'c4'
                          ? `${l.assigned_contributors_count ?? 0} / ${l.required_contributors ?? 3}`
                          : 'Standard'}
                      </span>
                    </td>
                    <td>
                      <span className="cell-mono">
                        {l.assessment_mode === 'c4'
                          ? `${l.processed_documents_count ?? 0} / ${l.required_contributors ?? 3} processed`
                          : 'Ready'}
                      </span>
                    </td>
                    <td>
                      <span className="cell-mono">
                        {l.submission_count ?? 0} ({l.completed_submission_count ?? 0} evaluated)
                      </span>
                    </td>
                    <td>
                      <span
                        className={`badge ${
                          l.readiness_status === 'ready_for_assessment' || l.readiness_status === 'ready'
                            ? 'badge-completed'
                            : l.readiness_status === 'extraction_failed'
                              ? 'badge-failed'
                              : 'badge-processing'
                        }`}
                      >
                        <span className="badge-dot" />
                        <span>{formatReadinessLabel(l.readiness_status)}</span>
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="table-card">
        <div className="table-card-header">
          <span className="table-card-title">Recent Submissions</span>
          <button className="btn btn-secondary btn-sm" onClick={() => onNavigate('submissions')}>
            View All
          </button>
        </div>

        {submissions.length === 0 ? (
          <div className="empty-state">
            <span className="empty-state-title">No student submissions yet</span>
            <p className="empty-state-description">Submitted student reflections will appear here.</p>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Student</th>
                  <th>Session</th>
                  <th>Submitted</th>
                  <th>Status</th>
                  <th>Score</th>
                  <th>Rating</th>
                  <th style={{ textAlign: 'right' }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {submissions.slice(0, 6).map(item => (
                  <tr key={item.id}>
                    <td>
                      <div className="cell-student-info">
                        <div className="cell-avatar">{item.student_name?.slice(0, 1).toUpperCase() || '?'}</div>
                        <div className="cell-student-text">
                          <span className="cell-student-name">{item.student_name || 'Student'}</span>
                          <span className="cell-student-email">{item.student_email}</span>
                        </div>
                      </div>
                    </td>
                    <td>
                      <span className="cell-lesson-title">{item.lesson}</span>
                    </td>
                    <td>
                      <span className="cell-mono">{new Date(item.submitted_at).toLocaleDateString()}</span>
                    </td>
                    <td>
                      <StatusBadge status={item.assessment_status} />
                    </td>
                    <td>
                      <span className="cell-score">
                        {item.assessment_mode === 'c4' && item.matched_concepts_count != null
                          ? `${item.matched_concepts_count}/${item.total_pool_concepts ?? 15}`
                          : item.recall_percentage !== null
                            ? `${item.recall_percentage}%`
                            : '—'}
                      </span>
                    </td>
                    <td>
                      {item.rating !== null ? (
                        <span className="cell-rating-badge">{item.rating}/5</span>
                      ) : (
                        <span style={{ color: 'var(--text-dim)' }}>—</span>
                      )}
                    </td>
                    <td>
                      <div className="cell-actions-group">
                        <button className="btn btn-secondary btn-sm" onClick={() => onSelectSubmission(item)}>
                          View
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}

function FacultyLessonsView({
  currentUser,
  lessons,
  submissions,
  students,
  request,
  onRefresh,
  onSuccess,
  onError,
  isContributorManager = false,
}: {
  currentUser?: User | null;
  lessons: Lesson[];
  submissions: Submission[];
  students: StudentOption[];
  request: (path: string, options?: RequestInit) => Promise<any>;
  onRefresh: () => Promise<void>;
  onSuccess: (msg: string) => void;
  onError: (msg: string) => void;
  isContributorManager?: boolean;
}) {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [form, setForm] = useState({
    title: '',
    group_name: '',
    session_date: '',
  });
  const [selectedContributors, setSelectedContributors] = useState<number[]>([]);
  const [studentSearch, setStudentSearch] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState('');

  // Per-session expanded details
  const [expandedLessonId, setExpandedLessonId] = useState<number | null>(null);
  const [contributorsMap, setContributorsMap] = useState<Record<number, LessonContributorsResponse>>({});
  const [poolMap, setPoolMap] = useState<Record<number, LessonKeywordPoolResponse>>({});
  const [loadingTelemetryId, setLoadingTelemetryId] = useState<number | null>(null);
  const [retryingDocId, setRetryingDocId] = useState<number | null>(null);
  const [editingContributorsForLesson, setEditingContributorsForLesson] = useState<number | null>(null);
  const [reassignIds, setReassignIds] = useState<number[]>([]);

  const eligibleContributors = useMemo(
    () => students.filter(s => s.role !== 'faculty'),
    [students]
  );

  const filteredStudents = useMemo(() => {
    if (!studentSearch.trim()) return eligibleContributors;
    const q = studentSearch.toLowerCase();
    return eligibleContributors.filter(
      s => s.name.toLowerCase().includes(q) || s.email.toLowerCase().includes(q)
    );
  }, [eligibleContributors, studentSearch]);

  const selectedStudentObjects = useMemo(
    () =>
      selectedContributors
        .map(id => eligibleContributors.find(s => s.id === id))
        .filter(Boolean) as StudentOption[],
    [selectedContributors, eligibleContributors]
  );

  const isCreateValid = useMemo(() => {
    if (!form.title.trim()) return false;
    if (selectedContributors.length !== 3) return false;
    return true;
  }, [form.title, selectedContributors]);

  function toggleContributorSelection(studentId: number) {
    setFormError('');
    setSelectedContributors(prev => {
      if (prev.includes(studentId)) {
        return prev.filter(id => id !== studentId);
      }
      if (prev.length >= 3) {
        return prev;
      }
      return [...prev, studentId];
    });
  }

  function resetCreateModal() {
    setForm({
      title: '',
      group_name: '',
      session_date: '',
    });
    setSelectedContributors([]);
    setFormError('');
    setStudentSearch('');
    setShowCreateModal(false);
  }

  async function loadC4Telemetry(lessonId: number) {
    setLoadingTelemetryId(lessonId);
    try {
      const [contribData, poolData] = await Promise.all([
        request(`/lessons/${lessonId}/contributors`),
        request(`/lessons/${lessonId}/keyword-pool`),
      ]);
      setContributorsMap(prev => ({ ...prev, [lessonId]: contribData }));
      setPoolMap(prev => ({ ...prev, [lessonId]: poolData }));
      if (contribData?.contributors) {
        setReassignIds(contribData.contributors.map((c: any) => c.student_id));
      }
    } catch (err: any) {
      onError(err.message || 'Failed to load session details');
    } finally {
      setLoadingTelemetryId(null);
    }
  }

  function handleToggleExpand(lesson: Lesson) {
    if (expandedLessonId === lesson.id) {
      setExpandedLessonId(null);
      setEditingContributorsForLesson(null);
      return;
    }
    setExpandedLessonId(lesson.id);
    setEditingContributorsForLesson(null);
    if (lesson.assessment_mode === 'c4') {
      loadC4Telemetry(lesson.id);
    }
  }

  async function handleCreate(e: FormEvent) {
    e.preventDefault();
    setFormError('');

    if (!form.title.trim()) {
      setFormError('Session title is required.');
      return;
    }
    if (selectedContributors.length !== 3) {
      setFormError('Please select exactly 3 eligible contributors.');
      return;
    }

    setSubmitting(true);
    try {
      const created = await request('/lessons', {
        method: 'POST',
        body: JSON.stringify({
          title: form.title.trim(),
          description: '',
          group_name: form.group_name.trim() || null,
          subject: SUBJECT,
          session_date: form.session_date,
          assessment_mode: 'c4',
          reference_text: '',
          concepts: [],
          contributor_ids: selectedContributors,
        }),
      });

      resetCreateModal();
      await onRefresh();
      onSuccess(`Created session "${created.title}". Contributors have been assigned.`);
      if (created.assessment_mode === 'c4') {
        setExpandedLessonId(created.id);
        await loadC4Telemetry(created.id);
      }
    } catch (err: any) {
      setFormError(err.message || 'Failed to create session');
    } finally {
      setSubmitting(false);
    }
  }

  async function handleSaveReassignedContributors(lessonId: number) {
    if (reassignIds.length !== 3) {
      onError('Exactly 3 contributors must be selected.');
      return;
    }
    try {
      await request(`/lessons/${lessonId}/contributors`, {
        method: 'POST',
        body: JSON.stringify({ student_ids: reassignIds }),
      });
      setEditingContributorsForLesson(null);
      await onRefresh();
      await loadC4Telemetry(lessonId);
      onSuccess('Contributors updated.');
    } catch (err: any) {
      onError(err.message || 'Failed to update contributors');
    }
  }

  async function handleFacultyRetryDocument(lessonId: number, documentId: number) {
    setRetryingDocId(documentId);
    try {
      await request(`/lessons/${lessonId}/reference-documents/${documentId}/retry`, {
        method: 'POST',
      });
      await onRefresh();
      await loadC4Telemetry(lessonId);
      onSuccess('Document re-processed.');
    } catch (err: any) {
      onError(err.message || 'Retry failed');
    } finally {
      setRetryingDocId(null);
    }
  }

  return (
    <>
      <div className="page-header">
        <div className="page-header-text">
          <h1 className="page-title">{isContributorManager ? 'Create & Manage Sessions' : 'Manage Sessions'}</h1>
          <p className="page-description">
            {isContributorManager
              ? 'Create collaborative Business Forecasting sessions, assign 3 contributors, and monitor report readiness.'
              : 'View published sessions, manage assigned contributors, or create a new session.'}
          </p>
        </div>
        <button
          className="btn btn-primary"
          onClick={() => {
            setFormError('');
            setShowCreateModal(true);
          }}
        >
          <Plus size={15} />
          <span>Create Session</span>
        </button>
      </div>

      {/* PUBLISHED SESSIONS LIST */}
      {lessons.length === 0 ? (
        <div className="table-card">
          <div className="empty-state">
            <span className="empty-state-title">No sessions managed yet</span>
            <p className="empty-state-description">Create your first Business Forecasting session to begin.</p>
            <button className="btn btn-primary btn-sm" onClick={() => setShowCreateModal(true)}>
              <Plus size={14} />
              <span>Create Session</span>
            </button>
          </div>
        </div>
      ) : (
        <div className="faculty-sessions-list">
          {lessons.map(lesson => {
            const isC4 = lesson.assessment_mode === 'c4';
            const isExpanded = expandedLessonId === lesson.id;
            const contribInfo = contributorsMap[lesson.id];
            const poolInfo = poolMap[lesson.id];
            const sessionSubs = submissions.filter(s => s.lesson_id === lesson.id);
            const completedSubs = sessionSubs.filter(s => s.assessment_status === 'completed');
            const completedDocsCount =
              contribInfo?.contributors.filter(c => c.processing_status === 'completed').length ?? 0;

            return (
              <div key={lesson.id} className="faculty-session-card">
                <div
                  className="faculty-session-card-top"
                  onClick={() => handleToggleExpand(lesson)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={e => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      handleToggleExpand(lesson);
                    }
                  }}
                >
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                      <span style={{ fontWeight: 600, fontSize: '14px', color: 'var(--text)' }}>{lesson.title}</span>
                      <span className="badge badge-draft">{isC4 ? 'C4 Pool' : 'Standard'}</span>
                      {lesson.creator_name && (
                        <span style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>
                          by {lesson.creator_name} ({lesson.creator_role || 'faculty'})
                        </span>
                      )}
                    </div>
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                      {lesson.session_date || 'No date'}
                      {lesson.description ? ` · ${lesson.description}` : ''} · {sessionSubs.length} submission(s) (
                      {completedSubs.length} evaluated)
                    </span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span
                      className={`badge ${
                        lesson.readiness_status === 'ready_for_assessment' || lesson.readiness_status === 'ready'
                          ? 'badge-completed'
                          : lesson.readiness_status === 'extraction_failed'
                            ? 'badge-failed'
                            : 'badge-processing'
                      }`}
                    >
                      <span className="badge-dot" />
                      <span>{formatReadinessLabel(lesson.readiness_status)}</span>
                    </span>
                    {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </div>
                </div>

                {isExpanded && (
                  <div className="faculty-session-expanded">
                    {!isC4 ? (
                      <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                        <strong>Reference Text:</strong>
                        <p style={{ marginTop: '4px', whiteSpace: 'pre-wrap' }}>{lesson.reference_text}</p>
                      </div>
                    ) : loadingTelemetryId === lesson.id ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 0' }}>
                        <Loader2 size={15} className="spin" />
                        <span style={{ fontSize: '12.5px', color: 'var(--text-muted)' }}>Loading session details...</span>
                      </div>
                    ) : (
                      <>
                        <div className="c4-progress-summary">
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                            <span>
                              Reports Processed: <strong>{completedDocsCount} / 3</strong>
                            </span>
                            <span>
                              Concept Pool:{' '}
                              <strong>
                                {poolInfo?.pool_ready
                                  ? `${poolInfo.distinct_concepts_count} distinct concepts`
                                  : 'Awaiting 3 reports'}
                              </strong>
                            </span>
                          </div>
                          <div className="char-meter" style={{ marginTop: '6px' }}>
                            <div
                              className="char-meter-fill"
                              style={{
                                width: `${(completedDocsCount / 3) * 100}%`,
                                background: completedDocsCount === 3 ? 'var(--success)' : 'var(--accent)',
                              }}
                            />
                          </div>
                        </div>

                        {poolInfo?.warning && (
                          <div className="c4-pool-warning-banner">
                            <AlertTriangle size={15} style={{ flexShrink: 0 }} />
                            <span>{poolInfo.warning}</span>
                          </div>
                        )}

                        <div>
                          <div
                            style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              marginBottom: '6px',
                            }}
                          >
                            <span style={{ fontSize: '12.5px', fontWeight: 600, color: 'var(--text)' }}>
                              Assigned Contributors ({contribInfo?.contributors.length || 0} / 3)
                            </span>
                            <button
                              type="button"
                              className="btn btn-secondary btn-sm"
                              onClick={() =>
                                setEditingContributorsForLesson(
                                  editingContributorsForLesson === lesson.id ? null : lesson.id
                                )
                              }
                            >
                              {editingContributorsForLesson === lesson.id ? 'Cancel' : 'Change Contributors'}
                            </button>
                          </div>

                          {editingContributorsForLesson === lesson.id && (
                            <div className="reassign-contributors-box">
                              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                                <span style={{ fontSize: '12px', fontWeight: 500 }}>Select 3 contributors:</span>
                                <span className="badge badge-draft">{reassignIds.length} / 3 selected</span>
                              </div>
                                <div className="student-selector-list" style={{ maxHeight: '150px' }}>
                                {eligibleContributors.map(st => {
                                  const checked = reassignIds.includes(st.id);
                                  const disabled = !checked && reassignIds.length >= 3;
                                  const isSelf = currentUser && st.id === currentUser.id;
                                  return (
                                    <label
                                      key={st.id}
                                      className={`student-selector-row ${checked ? 'selected' : ''} ${
                                        disabled ? 'disabled' : ''
                                      }`}
                                    >
                                      <input
                                        type="checkbox"
                                        checked={checked}
                                        disabled={disabled}
                                        onChange={() =>
                                          setReassignIds(prev =>
                                            prev.includes(st.id)
                                              ? prev.filter(x => x !== st.id)
                                              : prev.length < 3
                                                ? [...prev, st.id]
                                                : prev
                                          )
                                        }
                                      />
                                      <span style={{ fontSize: '12.5px', fontWeight: 500, color: 'var(--text)' }}>
                                        {st.name}
                                        {isSelf ? ' (You)' : ''}
                                      </span>
                                      <span style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>
                                        ({st.email})
                                      </span>
                                    </label>
                                  );
                                })}
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '8px' }}>
                                <button
                                  type="button"
                                  className="btn btn-primary btn-sm"
                                  disabled={reassignIds.length !== 3}
                                  onClick={() => handleSaveReassignedContributors(lesson.id)}
                                >
                                  Save Contributors
                                </button>
                              </div>
                            </div>
                          )}

                          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                            {contribInfo?.contributors.map((c, idx) => (
                              <div key={c.student_id} className="faculty-contributor-row">
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                  <div>
                                    <span style={{ fontSize: '12.5px', fontWeight: 600, color: 'var(--text)' }}>
                                      {idx + 1}. {c.student_name}
                                    </span>{' '}
                                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                                      ({c.student_email})
                                    </span>
                                  </div>
                                  <ContributorDocStatusBadge status={c.processing_status} />
                                </div>

                                {(c.report_title || c.original_filename) && (
                                  <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', marginTop: '3px' }}>
                                    {c.entry_mode === 'text' || c.entry_mode === 'direct_text'
                                      ? 'Written Report: '
                                      : 'Document: '}
                                    <strong>{c.report_title || c.original_filename}</strong>
                                  </div>
                                )}

                                {c.error_message && (
                                  <div
                                    style={{
                                      fontSize: '11.5px',
                                      color: 'var(--danger)',
                                      marginTop: '4px',
                                      display: 'flex',
                                      justifyContent: 'space-between',
                                      alignItems: 'center',
                                      gap: '8px',
                                    }}
                                  >
                                    <span>{c.error_message}</span>
                                    {c.document_id && (
                                      <button
                                        type="button"
                                        className="btn btn-secondary btn-sm"
                                        disabled={retryingDocId === c.document_id}
                                        onClick={() => handleFacultyRetryDocument(lesson.id, c.document_id!)}
                                      >
                                        {retryingDocId === c.document_id ? (
                                          <Loader2 size={12} className="spin" />
                                        ) : (
                                          <>
                                            <RotateCcw size={12} />
                                            <span>Retry</span>
                                          </>
                                        )}
                                      </button>
                                    )}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>

                        {poolInfo?.pool_ready && (
                          <div style={{ borderTop: '1px solid var(--border)', paddingTop: '10px' }}>
                            <span
                              style={{
                                fontSize: '12px',
                                fontWeight: 600,
                                color: 'var(--text)',
                                display: 'block',
                                marginBottom: '6px',
                              }}
                            >
                              Active Reference Pool ({poolInfo.distinct_concepts_count} concepts)
                            </span>
                            <div className="concept-tag-list">
                              {poolInfo.distinct_concepts.map((dc, i) => (
                                <span key={i} className="concept-pill">
                                  {dc.display_concept}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* CREATE SESSION MODAL */}
      {showCreateModal && (
        <div className="modal-center-backdrop" onClick={() => !submitting && resetCreateModal()}>
          <div className="modal-dialog" onClick={e => e.stopPropagation()} role="dialog" aria-modal="true">
            <div className="modal-dialog-header">
              <h2 className="modal-dialog-title">Create Session</h2>
              <button
                type="button"
                className="btn-icon"
                onClick={() => resetCreateModal()}
                disabled={submitting}
                aria-label="Close modal"
              >
                <X size={16} />
              </button>
            </div>

            <form onSubmit={handleCreate}>
              <div className="modal-dialog-body">
                {formError && (
                  <div className="form-error-alert" role="alert">
                    <AlertCircle size={15} />
                    <span>{formError}</span>
                  </div>
                )}

                {/* 1. Session Title */}
                <div className="form-group">
                  <label className="form-label" htmlFor="session-title">
                    Session Title
                  </label>
                  <input
                    id="session-title"
                    type="text"
                    required
                    placeholder="e.g., ARIMA Time Series & Seasonal Decomposition"
                    value={form.title}
                    onChange={e => setForm({ ...form, title: e.target.value })}
                  />
                </div>

                {/* 2. Session Date and Class / Group (Optional) */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  <div className="form-group">
                    <label className="form-label" htmlFor="session-date">
                      Session Date
                    </label>
                    <input
                      id="session-date"
                      type="date"
                      value={form.session_date}
                      onChange={e => setForm({ ...form, session_date: e.target.value })}
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label" htmlFor="session-group">
                      Class / Group (Optional)
                    </label>
                    <input
                      id="session-group"
                      type="text"
                      placeholder="e.g., Section A"
                      value={form.group_name}
                      onChange={e => setForm({ ...form, group_name: e.target.value })}
                    />
                  </div>
                </div>

                {/* 3. Assign 3 Contributors */}
                <div className="form-group">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <label className="form-label">Assign 3 Contributors</label>
                    <span
                      className={`badge ${
                        selectedContributors.length === 3 ? 'badge-completed' : 'badge-pending'
                      }`}
                    >
                      {selectedContributors.length} of 3 selected
                    </span>
                  </div>
                  <span style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>
                    Select the 3 contributors who will each submit 1 reference report. Session creators are not
                    counted as contributors unless explicitly selected here.
                  </span>

                  {selectedStudentObjects.length > 0 && (
                    <div className="selected-contributors-review">
                      {selectedStudentObjects.map(st => {
                        const isSelf = currentUser && st.id === currentUser.id;
                        return (
                          <span key={st.id} className="contributor-selected-chip">
                            <span>
                              {st.name}
                              {isSelf ? ' (You)' : ''}
                            </span>
                            <button
                              type="button"
                              onClick={() => toggleContributorSelection(st.id)}
                              aria-label={`Remove ${st.name}`}
                            >
                              <X size={12} />
                            </button>
                          </span>
                        );
                      })}
                    </div>
                  )}

                  <div className="input-with-icon" style={{ marginTop: '4px' }}>
                    <Search size={14} />
                    <input
                      type="text"
                      placeholder="Search contributors by name or email..."
                      value={studentSearch}
                      onChange={e => setStudentSearch(e.target.value)}
                    />
                  </div>

                  <div className="student-selector-list" style={{ marginTop: '4px', maxHeight: '160px' }}>
                    {filteredStudents.length === 0 ? (
                      <div style={{ padding: '12px', fontSize: '12px', color: 'var(--text-muted)', textAlign: 'center' }}>
                        No matching accounts found.
                      </div>
                    ) : (
                      filteredStudents.map(st => {
                        const isChecked = selectedContributors.includes(st.id);
                        const isDisabled = !isChecked && selectedContributors.length >= 3;
                        const isSelf = currentUser && st.id === currentUser.id;
                        return (
                          <label
                            key={st.id}
                            className={`student-selector-row ${isChecked ? 'selected' : ''} ${
                              isDisabled ? 'disabled' : ''
                            }`}
                          >
                            <input
                              type="checkbox"
                              checked={isChecked}
                              disabled={isDisabled}
                              onChange={() => toggleContributorSelection(st.id)}
                            />
                            <div style={{ display: 'flex', justifyContent: 'space-between', flex: 1, gap: '8px' }}>
                              <div>
                                <div style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text)' }}>
                                  {st.name}
                                  {isSelf ? ' (You — Creator)' : ''}
                                </div>
                                <div style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>{st.email}</div>
                              </div>
                              <span className="badge badge-draft" style={{ alignSelf: 'center' }}>
                                {st.role}
                              </span>
                            </div>
                          </label>
                        );
                      })
                    )}
                  </div>
                </div>
              </div>

              {/* Modal Actions */}
              <div className="modal-dialog-footer">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => resetCreateModal()}
                  disabled={submitting}
                >
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary" disabled={submitting || !isCreateValid}>
                  {submitting ? (
                    <>
                      <Loader2 size={14} className="spin" />
                      <span>Creating...</span>
                    </>
                  ) : (
                    <span>Create Session</span>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
}

// ============================================================================
// 5. CONTRIBUTOR DASHBOARD — OVERVIEW, SUBMIT REFERENCE REPORT (UPLOAD OR WRITE), WRITE REFLECTION, MANAGE SESSIONS, MY SUBMISSIONS
// ============================================================================
function formatContributorDocStatusText(status?: string): string {
  switch (status) {
    case 'completed':
      return 'Processed';
    case 'uploading':
    case 'processing':
      return 'Processing';
    case 'uploaded':
      return 'Submitted';
    case 'failed':
      return 'Failed';
    default:
      return 'Pending';
  }
}

function ContributorPortalWorkspace({
  currentUser,
  view,
  setView,
  assignments,
  uploads,
  lessons,
  submissions,
  students,
  analytics,
  loading,
  preselectedLessonId,
  setPreselectedLessonId,
  preselectedReflectionLessonId,
  setPreselectedReflectionLessonId,
  onSelectSubmission,
  onDownload,
  request,
  onRefresh,
  onSuccess,
  onError,
}: {
  currentUser?: User | null;
  view: string;
  setView: (v: string) => void;
  assignments: ContributorAssignmentItem[];
  uploads: ContributorUploadHistoryItem[];
  lessons: Lesson[];
  submissions: Submission[];
  students: StudentOption[];
  analytics: DashboardAnalyticsResponse | null;
  loading: boolean;
  preselectedLessonId: number | null;
  setPreselectedLessonId: (id: number | null) => void;
  preselectedReflectionLessonId: number | null;
  setPreselectedReflectionLessonId: (id: number | null) => void;
  onSelectSubmission: (sub: Submission) => void;
  onDownload: (path: string, name: string) => Promise<void>;
  request: (path: string, options?: RequestInit) => Promise<any>;
  onRefresh: () => Promise<void>;
  onSuccess: (msg: string) => void;
  onError: (msg: string) => void;
}) {
  const [selectedLessonId, setSelectedLessonId] = useState<number | null>(() => {
    return preselectedLessonId || assignments[0]?.lesson_id || null;
  });
  const [entryMode, setEntryMode] = useState<'file_upload' | 'direct_text'>('file_upload');
  const [reportTitle, setReportTitle] = useState<string>('');
  const [reportText, setReportText] = useState<string>('');
  const [loadingOwnReport, setLoadingOwnReport] = useState<boolean>(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploadingLessonId, setUploadingLessonId] = useState<number | null>(null);
  const [retryingLessonId, setRetryingLessonId] = useState<number | null>(null);
  const [fileError, setFileError] = useState<string>('');
  const [uploadStatusBanner, setUploadStatusBanner] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);
  const [submissionTypeFilter, setSubmissionTypeFilter] = useState<'all' | 'document' | 'reflection'>('all');
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (preselectedLessonId) {
      setSelectedLessonId(preselectedLessonId);
    } else if (!selectedLessonId && assignments.length > 0) {
      setSelectedLessonId(assignments[0].lesson_id);
    }
  }, [preselectedLessonId, assignments]);

  const activeAssignment =
    assignments.find(a => a.lesson_id === selectedLessonId) || assignments[0] || null;

  useEffect(() => {
    let cancelled = false;
    async function loadMyReportForLesson(lessonId: number) {
      setLoadingOwnReport(true);
      try {
        const res = await request(`/lessons/${lessonId}/reference-documents/me`);
        if (cancelled) return;
        if (res?.document_uploaded && (res.entry_mode === 'text' || res.entry_mode === 'direct_text')) {
          setReportTitle(res.report_title || '');
          setReportText(res.report_text || res.content || '');
          setEntryMode('direct_text');
        } else if (res?.document_uploaded && res.entry_mode === 'file') {
          setReportTitle('');
          setReportText('');
          setEntryMode('file_upload');
        } else {
          setReportTitle('');
          setReportText('');
        }
      } catch {
        if (!cancelled) {
          setReportTitle('');
          setReportText('');
        }
      } finally {
        if (!cancelled) setLoadingOwnReport(false);
      }
    }
    if (view === 'submit-document' && activeAssignment?.lesson_id) {
      loadMyReportForLesson(activeAssignment.lesson_id);
    }
    return () => {
      cancelled = true;
    };
  }, [view, activeAssignment?.lesson_id]);

  function validateAndSelectFile(file: File | null) {
    setFileError('');
    setUploadStatusBanner(null);
    if (!file) {
      setSelectedFile(null);
      return;
    }
    const lowerName = file.name.toLowerCase();
    const hasValidExt = ALLOWED_UPLOAD_EXTENSIONS.some(ext => lowerName.endsWith(ext));
    if (!hasValidExt) {
      setFileError(`Unsupported file format. Accepted formats: ${ALLOWED_UPLOAD_EXTENSIONS.join(', ').toUpperCase()}`);
      setSelectedFile(null);
      return;
    }
    if (file.size <= 0) {
      setFileError('Selected file is empty (0 bytes).');
      setSelectedFile(null);
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setFileError(`File exceeds the 10 MB size limit (${formatFileSize(file.size)}).`);
      setSelectedFile(null);
      return;
    }
    setSelectedFile(file);
  }

  async function handleUploadDocument(targetLessonId: number) {
    if (!selectedFile || uploadingLessonId !== null) return;
    setUploadingLessonId(targetLessonId);
    setFileError('');
    setUploadStatusBanner(null);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      const res = await request(`/lessons/${targetLessonId}/reference-documents`, {
        method: 'POST',
        body: formData,
      });
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      await onRefresh();
      if (res.processing_status === 'completed') {
        const msg = `Document "${res.original_filename}" uploaded and processed successfully.`;
        setUploadStatusBanner({ type: 'success', message: msg });
        onSuccess(msg);
      } else if (res.processing_status === 'failed') {
        const errMsg = res.error_message || 'Document uploaded, but keyword extraction failed. You may retry.';
        setUploadStatusBanner({ type: 'error', message: errMsg });
        onError(errMsg);
      } else {
        const msg = `Document "${res.original_filename}" uploaded (status: ${formatContributorDocStatusText(res.processing_status)}).`;
        setUploadStatusBanner({ type: 'success', message: msg });
        onSuccess(msg);
      }
    } catch (err: any) {
      const errMsg = err.message || 'Document upload failed.';
      setUploadStatusBanner({ type: 'error', message: errMsg });
      onError(errMsg);
    } finally {
      setUploadingLessonId(null);
    }
  }

  async function handleSubmitWrittenReport(targetLessonId: number) {
    if (uploadingLessonId !== null) return;
    setFileError('');
    setUploadStatusBanner(null);
    if (!targetLessonId) {
      setFileError('Please select an assigned session first.');
      return;
    }
    const trimmedText = reportText.trim();
    if (!trimmedText) {
      setFileError('Reference report content is required and cannot be empty.');
      return;
    }
    const words = trimmedText.split(/\s+/).filter(Boolean);
    if (trimmedText.length < 20 || words.length < 3) {
      setFileError('Reference report is too short (please enter at least 20 characters and 3 words).');
      return;
    }
    const finalTitle =
      reportTitle.trim() || `Reference Report (${activeAssignment?.lesson_title || 'Session'})`;
    setUploadingLessonId(targetLessonId);
    try {
      const res = await request(`/lessons/${targetLessonId}/reference-reports`, {
        method: 'POST',
        body: JSON.stringify({
          title: finalTitle,
          content: trimmedText,
          report_text: trimmedText,
          lesson_id: targetLessonId,
        }),
      });
      if (res?.report_title) {
        setReportTitle(res.report_title);
      } else {
        setReportTitle(finalTitle);
      }
      await onRefresh();
      if (res.processing_status === 'completed') {
        const msg = `Reference report "${res.report_title || finalTitle}" saved and processed successfully.`;
        setUploadStatusBanner({ type: 'success', message: msg });
        onSuccess(msg);
      } else if (res.processing_status === 'failed') {
        const errMsg = res.error_message || 'Report saved, but keyword extraction failed. You may retry.';
        setUploadStatusBanner({ type: 'error', message: errMsg });
        onError(errMsg);
      } else {
        const msg = `Reference report saved (status: ${formatContributorDocStatusText(res.processing_status)}).`;
        setUploadStatusBanner({ type: 'success', message: msg });
        onSuccess(msg);
      }
    } catch (err: any) {
      // Preserve written text and title on error
      const errMsg = err.message || 'Failed to save reference report.';
      setFileError(errMsg);
      setUploadStatusBanner({ type: 'error', message: errMsg });
      onError(errMsg);
    } finally {
      setUploadingLessonId(null);
    }
  }

  async function handleRetryDocument(targetLessonId: number) {
    if (retryingLessonId !== null) return;
    setRetryingLessonId(targetLessonId);
    setUploadStatusBanner(null);
    try {
      const res = await request(`/lessons/${targetLessonId}/reference-documents/me/retry`, {
        method: 'POST',
      });
      await onRefresh();
      if (res.processing_status === 'completed') {
        const msg = 'Reference report processed successfully.';
        setUploadStatusBanner({ type: 'success', message: msg });
        onSuccess(msg);
      } else {
        const errMsg = res.error_message || 'Retry failed. Please try again or submit a replacement report.';
        setUploadStatusBanner({ type: 'error', message: errMsg });
        onError(errMsg);
      }
    } catch (err: any) {
      const errMsg = err.message || 'Retry failed.';
      setUploadStatusBanner({ type: 'error', message: errMsg });
      onError(errMsg);
    } finally {
      setRetryingLessonId(null);
    }
  }

  const uploadedCount = assignments.filter(a => a.document_uploaded).length;
  const completedCount = assignments.filter(a => a.processing_status === 'completed').length;
  const managedLessons = useMemo(
    () => lessons.filter(l => l.is_creator || l.can_manage),
    [lessons]
  );
  const reportWordCount = useMemo(() => {
    const t = reportText.trim();
    return t ? t.split(/\s+/).length : 0;
  }, [reportText]);

  // --------------------------------------------------------------------------
  // SECTION 5: MANAGE SESSIONS (CREATE SESSION & MONITOR MANAGED SESSIONS)
  // --------------------------------------------------------------------------
  if (view === 'manage-sessions') {
    return (
      <FacultyLessonsView
        currentUser={currentUser}
        isContributorManager={true}
        lessons={managedLessons}
        submissions={submissions}
        students={students}
        request={request}
        onRefresh={onRefresh}
        onSuccess={onSuccess}
        onError={onError}
      />
    );
  }

  // --------------------------------------------------------------------------
  // SECTION 2: SUBMIT REFERENCE REPORT (UPLOAD DOCUMENT OR WRITE REPORT)
  // --------------------------------------------------------------------------
  if (view === 'submit-document') {
    const isUploading = activeAssignment ? uploadingLessonId === activeAssignment.lesson_id : false;
    const isRetrying = activeAssignment ? retryingLessonId === activeAssignment.lesson_id : false;
    const currentStatus = isUploading
      ? 'processing'
      : activeAssignment?.document_uploaded
        ? activeAssignment.processing_status
        : 'awaiting_upload';

    return (
      <>
        <div className="page-header">
          <div className="page-header-text">
            <h1 className="page-title">Submit Reference Report</h1>
            <p className="page-description">
              Select an assigned session and either upload a reference document (.PDF, .DOCX, .TXT) or write/paste your
              reference report directly.
            </p>
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <button className="btn btn-secondary" onClick={() => setView('my-submissions')}>
              <span>My Submissions</span>
            </button>
            <button className="btn btn-secondary" onClick={() => setView('dashboard')}>
              <span>Overview</span>
            </button>
          </div>
        </div>

        <div className="upload-form-container">
          <div className="editor-card">
            <div className="form-group">
              <label className="form-label" htmlFor="contributor-session-select">
                Assigned Session
              </label>
              <select
                id="contributor-session-select"
                value={activeAssignment?.lesson_id || ''}
                onChange={e => {
                  const nextId = e.target.value ? Number(e.target.value) : null;
                  setSelectedLessonId(nextId);
                  setPreselectedLessonId(nextId);
                  setSelectedFile(null);
                  setFileError('');
                  setUploadStatusBanner(null);
                  if (fileInputRef.current) fileInputRef.current.value = '';
                }}
                disabled={isUploading || isRetrying || assignments.length === 0}
              >
                {assignments.length === 0 ? (
                  <option value="">-- No assigned sessions available --</option>
                ) : (
                  assignments.map(a => (
                    <option key={a.lesson_id} value={a.lesson_id}>
                      {a.lesson_title} — {formatContributorDocStatusText(a.processing_status)}
                    </option>
                  ))
                )}
              </select>
            </div>

            {/* REAL BACKEND PROCESSING STATUS PANEL */}
            {activeAssignment && (
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '8px',
                  padding: '12px 14px',
                  backgroundColor: 'var(--bg-subtle)',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: '12.5px', fontWeight: 600, color: 'var(--text)' }}>
                    Reference Report Processing Status
                  </span>
                  <ContributorDocStatusBadge status={currentStatus} />
                </div>

                <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>
                  {activeAssignment.document_uploaded ? (
                    <>
                      {activeAssignment.entry_mode === 'text' || activeAssignment.entry_mode === 'direct_text'
                        ? 'Written report: '
                        : 'Submitted file: '}
                      <strong>{activeAssignment.report_title || activeAssignment.original_filename}</strong>
                      {activeAssignment.file_size ? ` (${formatFileSize(activeAssignment.file_size)})` : ''}
                      {activeAssignment.version ? ` · v${activeAssignment.version}` : ''}
                      {activeAssignment.uploaded_at
                        ? ` · Submitted ${new Date(activeAssignment.uploaded_at).toLocaleString()}`
                        : ''}
                    </>
                  ) : (
                    <span>
                      Status: Pending. Choose Upload Document or Write Report below to submit your reference material.
                    </span>
                  )}
                </div>

                {activeAssignment.processing_status === 'completed' && !isUploading && (
                  <div style={{ fontSize: '12px', color: 'var(--success)', fontWeight: 500 }}>
                    Processed successfully by NVIDIA Nemotron. Session readiness:{' '}
                    {formatReadinessLabel(activeAssignment.session_readiness_status)}.
                  </div>
                )}

                {activeAssignment.processing_status === 'failed' && !isUploading && (
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      gap: '10px',
                      marginTop: '4px',
                      paddingTop: '8px',
                      borderTop: '1px solid var(--border)',
                    }}
                  >
                    <span style={{ fontSize: '12px', color: 'var(--danger)' }}>
                      {activeAssignment.error_message || 'Reference report keyword extraction failed.'}
                    </span>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => handleRetryDocument(activeAssignment.lesson_id)}
                      disabled={isRetrying || isUploading}
                    >
                      {isRetrying ? (
                        <>
                          <Loader2 size={12} className="spin" />
                          <span>Retrying...</span>
                        </>
                      ) : (
                        <>
                          <RotateCcw size={12} />
                          <span>Retry Processing</span>
                        </>
                      )}
                    </button>
                  </div>
                )}
              </div>
            )}

            {uploadStatusBanner && (
              <div
                className={uploadStatusBanner.type === 'error' ? 'form-error-alert' : 'form-success-alert'}
                role="alert"
              >
                {uploadStatusBanner.type === 'error' ? <AlertCircle size={15} /> : <CheckCircle2 size={15} />}
                <span>{uploadStatusBanner.message}</span>
              </div>
            )}

            {/* SEGMENTED CONTROL: UPLOAD DOCUMENT vs WRITE REPORT */}
            <div className="form-group">
              <label className="form-label">Entry Method</label>
              <div className="segmented-control" role="tablist" aria-label="Reference report entry method">
                <button
                  type="button"
                  role="tab"
                  aria-selected={entryMode === 'file_upload'}
                  className={`segmented-tab ${entryMode === 'file_upload' ? 'active' : ''}`}
                  onClick={() => {
                    setEntryMode('file_upload');
                    setFileError('');
                    setUploadStatusBanner(null);
                  }}
                >
                  <Upload size={14} />
                  <span>Upload Document</span>
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={entryMode === 'direct_text'}
                  className={`segmented-tab ${entryMode === 'direct_text' ? 'active' : ''}`}
                  onClick={() => {
                    setEntryMode('direct_text');
                    setFileError('');
                    setUploadStatusBanner(null);
                  }}
                >
                  <PenLine size={14} />
                  <span>Write Report</span>
                </button>
              </div>
            </div>

            {entryMode === 'file_upload' ? (
              <>
                {/* OPTION A: UPLOAD A DOCUMENT FILE */}
                <div className="form-group">
                  <label className="form-label">Reference Document File (.PDF, .DOCX, .TXT · Max 10 MB)</label>
                  <div
                    className={`dropzone-box ${dragActive ? 'drag-active' : ''} ${selectedFile ? 'has-file' : ''}`}
                    onDragOver={e => {
                      e.preventDefault();
                      e.stopPropagation();
                      if (!isUploading && !isRetrying && activeAssignment) setDragActive(true);
                    }}
                    onDragLeave={e => {
                      e.preventDefault();
                      e.stopPropagation();
                      setDragActive(false);
                    }}
                    onDrop={e => {
                      e.preventDefault();
                      e.stopPropagation();
                      setDragActive(false);
                      if (!isUploading && !isRetrying && activeAssignment) {
                        validateAndSelectFile(e.dataTransfer.files?.[0] || null);
                      }
                    }}
                    onClick={() => {
                      if (!isUploading && !isRetrying && activeAssignment) {
                        fileInputRef.current?.click();
                      }
                    }}
                    role="button"
                    tabIndex={0}
                    onKeyDown={e => {
                      if ((e.key === 'Enter' || e.key === ' ') && !isUploading && !isRetrying && activeAssignment) {
                        e.preventDefault();
                        fileInputRef.current?.click();
                      }
                    }}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
                      onChange={e => validateAndSelectFile(e.target.files?.[0] || null)}
                      style={{ display: 'none' }}
                      disabled={isUploading || isRetrying || !activeAssignment}
                    />
                    <div className="dropzone-icon">
                      <Upload size={18} />
                    </div>
                    {selectedFile ? (
                      <>
                        <div className="dropzone-title">{selectedFile.name}</div>
                        <div className="dropzone-sub">
                          Size: {formatFileSize(selectedFile.size)} · Click to choose a different file
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="dropzone-title">Click to select a file or drag and drop here</div>
                        <div className="dropzone-sub">Accepted formats: .PDF, .DOCX, .TXT · Maximum size: 10 MB</div>
                      </>
                    )}
                  </div>
                </div>

                {selectedFile && (
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '8px 12px',
                      backgroundColor: 'var(--bg-subtle)',
                      borderRadius: 'var(--radius-sm)',
                      border: '1px solid var(--border)',
                      fontSize: '12.5px',
                    }}
                  >
                    <span>
                      Selected file: <strong>{selectedFile.name}</strong> ({formatFileSize(selectedFile.size)})
                    </span>
                    <button
                      type="button"
                      className="btn btn-ghost btn-sm"
                      disabled={isUploading}
                      onClick={() => {
                        setSelectedFile(null);
                        if (fileInputRef.current) fileInputRef.current.value = '';
                      }}
                    >
                      Clear
                    </button>
                  </div>
                )}

                {fileError && (
                  <div className="form-error-alert" role="alert">
                    <AlertCircle size={15} />
                    <span>{fileError}</span>
                  </div>
                )}

                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
                  <button
                    type="button"
                    className="btn btn-primary"
                    disabled={!activeAssignment || !selectedFile || isUploading || isRetrying}
                    onClick={() => activeAssignment && handleUploadDocument(activeAssignment.lesson_id)}
                  >
                    {isUploading ? (
                      <>
                        <Loader2 size={14} className="spin" />
                        <span>Uploading & Processing...</span>
                      </>
                    ) : (
                      <>
                        <Upload size={14} />
                        <span>{activeAssignment?.document_uploaded ? 'Replace Document' : 'Upload Document'}</span>
                      </>
                    )}
                  </button>
                </div>
              </>
            ) : (
              <>
                {/* OPTION B: WRITE OR PASTE REPORT DIRECTLY */}
                <div className="form-group">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <label className="form-label" htmlFor="direct-report-title">
                      Report Title
                    </label>
                    {loadingOwnReport && (
                      <span style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>Loading saved report...</span>
                    )}
                  </div>
                  <input
                    id="direct-report-title"
                    className="direct-report-title-input"
                    type="text"
                    placeholder="Report Title (e.g., ARIMA & Seasonal Decomposition Notes)"
                    value={reportTitle}
                    onChange={e => {
                      setReportTitle(e.target.value);
                      if (fileError) setFileError('');
                    }}
                    disabled={isUploading || isRetrying || !activeAssignment}
                  />
                </div>

                <div className="form-group">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <label className="form-label" htmlFor="direct-report-textarea">
                      Report Content
                    </label>
                    <span className="char-counter-text">
                      {reportWordCount} words · {reportText.length} characters
                    </span>
                  </div>
                  <textarea
                    id="direct-report-textarea"
                    className="direct-report-textarea"
                    rows={10}
                    placeholder="Write or paste your full Business Forecasting reference report here..."
                    value={reportText}
                    onChange={e => {
                      setReportText(e.target.value);
                      if (fileError) setFileError('');
                    }}
                    disabled={isUploading || isRetrying || !activeAssignment}
                  />
                </div>

                {fileError && (
                  <div className="form-error-alert" role="alert">
                    <AlertCircle size={15} />
                    <span>{fileError}</span>
                  </div>
                )}

                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
                  <button
                    type="button"
                    className="btn btn-primary"
                    disabled={
                      !activeAssignment ||
                      reportText.trim().length === 0 ||
                      isUploading ||
                      isRetrying
                    }
                    onClick={() => activeAssignment && handleSubmitWrittenReport(activeAssignment.lesson_id)}
                  >
                    {isUploading ? (
                      <>
                        <Loader2 size={14} className="spin" />
                        <span>Saving & Extracting Concepts...</span>
                      </>
                    ) : (
                      <>
                        <PenLine size={14} />
                        <span>Save and Submit Report</span>
                      </>
                    )}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </>
    );
  }

  // --------------------------------------------------------------------------
  // SECTION 3: WRITE AND SUBMIT REFLECTION VIEW
  // --------------------------------------------------------------------------
  if (view === 'write-reflection') {
    return (
      <ReflectionSubmitView
        lessons={lessons}
        submissions={submissions}
        initialLessonId={preselectedReflectionLessonId || activeAssignment?.lesson_id || null}
        request={request}
        onSelectSubmission={onSelectSubmission}
        onDownload={onDownload}
        onDone={async () => {
          await onRefresh();
          onSuccess('Reflection submitted and saved.');
        }}
        onError={msg => onError(msg)}
      />
    );
  }

  // --------------------------------------------------------------------------
  // SECTION 4: MY SUBMISSIONS (DOCUMENTS/REPORTS & REFLECTIONS)
  // --------------------------------------------------------------------------
  if (view === 'my-submissions') {
    const showDocs = submissionTypeFilter === 'all' || submissionTypeFilter === 'document';
    const showReflections = submissionTypeFilter === 'all' || submissionTypeFilter === 'reflection';

    type CombinedRow =
      | {
          kind: 'document';
          key: string;
          timestamp: string;
          doc: ContributorUploadHistoryItem;
        }
      | {
          kind: 'reflection';
          key: string;
          timestamp: string;
          sub: Submission;
        };

    const combinedRows: CombinedRow[] = [
      ...(showDocs
        ? uploads.map(doc => ({
            kind: 'document' as const,
            key: `doc-${doc.id}`,
            timestamp: doc.uploaded_at,
            doc,
          }))
        : []),
      ...(showReflections
        ? submissions.map(sub => ({
            kind: 'reflection' as const,
            key: `sub-${sub.id}`,
            timestamp: sub.submitted_at,
            sub,
          }))
        : []),
    ].sort((a, b) => (a.timestamp < b.timestamp ? 1 : -1));

    return (
      <>
        <div className="page-header">
          <div className="page-header-text">
            <h1 className="page-title">My Submissions</h1>
            <p className="page-description">
              View your submitted reference reports (uploaded documents or written reports) and reflections.
            </p>
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <button className="btn btn-secondary" onClick={() => setView('submit-document')}>
              <Upload size={14} />
              <span>Submit Reference Report</span>
            </button>
            <button className="btn btn-primary" onClick={() => setView('write-reflection')}>
              <Send size={14} />
              <span>Write Reflection</span>
            </button>
          </div>
        </div>

        <div className="filter-card">
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
            <button
              type="button"
              className={`btn ${submissionTypeFilter === 'all' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
              onClick={() => setSubmissionTypeFilter('all')}
            >
              All ({uploads.length + submissions.length})
            </button>
            <button
              type="button"
              className={`btn ${submissionTypeFilter === 'document' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
              onClick={() => setSubmissionTypeFilter('document')}
            >
              Reference Reports ({uploads.length})
            </button>
            <button
              type="button"
              className={`btn ${submissionTypeFilter === 'reflection' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
              onClick={() => setSubmissionTypeFilter('reflection')}
            >
              Reflections ({submissions.length})
            </button>
          </div>
        </div>

        <div className="table-card">
          {combinedRows.length === 0 ? (
            <div className="empty-state">
              <span className="empty-state-title">No submissions yet</span>
              <p className="empty-state-description">
                Submitted reference reports and reflections will appear here.
              </p>
            </div>
          ) : (
            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Session Name</th>
                    <th>Submission Type</th>
                    <th>Submission Date</th>
                    <th>Status</th>
                    <th>Rating & Feedback</th>
                    <th style={{ textAlign: 'right' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {combinedRows.map(row => {
                    if (row.kind === 'document') {
                      const doc = row.doc;
                      const isRetrying = retryingLessonId === doc.lesson_id;
                      const isDirect = doc.entry_mode === 'text' || doc.entry_mode === 'direct_text';
                      return (
                        <tr key={row.key}>
                          <td>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                              <span className="cell-lesson-title">{doc.lesson_title}</span>
                              <span style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>
                                {doc.report_title || doc.original_filename} ({formatFileSize(doc.file_size)}) · v
                                {doc.version}
                                {!doc.is_active ? ' (Replaced)' : ''}
                              </span>
                            </div>
                          </td>
                          <td>
                            <span className="badge badge-draft">
                              {isDirect ? 'Written Report' : 'Uploaded Document'}
                            </span>
                          </td>
                          <td>
                            <span className="cell-mono">{new Date(doc.uploaded_at).toLocaleDateString()}</span>
                          </td>
                          <td>
                            <ContributorDocStatusBadge status={doc.processing_status} />
                          </td>
                          <td>
                            {doc.error_message ? (
                              <span style={{ fontSize: '12px', color: 'var(--danger)' }}>{doc.error_message}</span>
                            ) : doc.processing_status === 'completed' ? (
                              <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                                Reference report processed
                              </span>
                            ) : (
                              <span style={{ color: 'var(--text-dim)' }}>—</span>
                            )}
                          </td>
                          <td>
                            <div className="cell-actions-group">
                              {doc.is_active && doc.processing_status === 'failed' && (
                                <button
                                  type="button"
                                  className="btn btn-secondary btn-sm"
                                  disabled={isRetrying}
                                  onClick={() => handleRetryDocument(doc.lesson_id)}
                                >
                                  {isRetrying ? (
                                    <Loader2 size={12} className="spin" />
                                  ) : (
                                    <>
                                      <RotateCcw size={12} />
                                      <span>Retry</span>
                                    </>
                                  )}
                                </button>
                              )}
                              <button
                                type="button"
                                className="btn btn-secondary btn-sm"
                                onClick={() => {
                                  setPreselectedLessonId(doc.lesson_id);
                                  setSelectedLessonId(doc.lesson_id);
                                  setEntryMode(isDirect ? 'direct_text' : 'file_upload');
                                  setView('submit-document');
                                }}
                              >
                                {doc.processing_status === 'failed' ? 'Replace' : isDirect ? 'Edit / Replace' : 'View / Replace'}
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    }

                    const sub = row.sub;
                    return (
                      <tr key={row.key}>
                        <td>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                            <span className="cell-lesson-title">{sub.lesson}</span>
                            <span style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>
                              {sub.character_count} / 500 characters
                            </span>
                          </div>
                        </td>
                        <td>
                          <span className="badge badge-completed">Reflection</span>
                        </td>
                        <td>
                          <span className="cell-mono">{new Date(sub.submitted_at).toLocaleDateString()}</span>
                        </td>
                        <td>
                          <StatusBadge status={sub.assessment_status} />
                        </td>
                        <td>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '3px', maxWidth: '300px' }}>
                            {sub.rating !== null ? (
                              <span className="cell-rating-badge" style={{ width: 'fit-content' }}>
                                <Star size={11} fill="currentColor" />
                                <span>{sub.rating} / 5</span>
                              </span>
                            ) : (
                              <span style={{ fontSize: '12px', color: 'var(--text-dim)' }}>Pending rating</span>
                            )}
                            {sub.feedback && (
                              <span
                                style={{
                                  fontSize: '11.5px',
                                  color: 'var(--text-secondary)',
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                  whiteSpace: 'nowrap',
                                }}
                                title={sub.feedback}
                              >
                                {sub.feedback}
                              </span>
                            )}
                          </div>
                        </td>
                        <td>
                          <div className="cell-actions-group">
                            <button
                              type="button"
                              className="btn btn-secondary btn-sm"
                              onClick={() => onSelectSubmission(sub)}
                            >
                              View
                            </button>
                            {sub.assessment_status === 'completed' && (
                              <button
                                type="button"
                                className="btn-icon"
                                onClick={() =>
                                  onDownload(`/submissions/${sub.id}/report.pdf`, `recallai-report-${sub.id}.pdf`)
                                }
                                title="Download PDF Report"
                              >
                                <Download size={14} />
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </>
    );
  }

  // --------------------------------------------------------------------------
  // SECTION 1: CONTRIBUTOR OVERVIEW (DEFAULT DASHBOARD VIEW)
  // --------------------------------------------------------------------------
  return (
    <>
      <div className="page-header">
        <div className="page-header-text">
          <h1 className="page-title">Contributor Dashboard</h1>
          <p className="page-description">
            Create sessions, submit reference reports (upload document or write report), and track assessment readiness.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <button className="btn btn-secondary" onClick={() => setView('manage-sessions')}>
            <Plus size={14} />
            <span>Create Session</span>
          </button>
          <button className="btn btn-secondary" onClick={() => setView('write-reflection')}>
            <Send size={14} />
            <span>Write Reflection</span>
          </button>
          <button className="btn btn-primary" onClick={() => setView('submit-document')}>
            <Upload size={15} />
            <span>Submit Reference Report</span>
          </button>
        </div>
      </div>

      <div className="kpi-grid">
        <div className="kpi-card">
          <span className="kpi-label">Assigned Sessions</span>
          <span className="kpi-value">{loading ? '—' : assignments.length}</span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Reports Submitted</span>
          <span className="kpi-value">
            {loading ? '—' : `${uploadedCount} / ${assignments.length}`}
          </span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Processed Successfully</span>
          <span className="kpi-value">{loading ? '—' : completedCount}</span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Sessions Created by You</span>
          <span className="kpi-value">{loading ? '—' : managedLessons.length}</span>
        </div>
      </div>

      {/* ASSIGNED SESSIONS FOR REFERENCE REPORT SUBMISSION */}
      <div className="table-card">
        <div className="table-card-header">
          <span className="table-card-title">Assigned Sessions — Reference Report Status</span>
        </div>

        {assignments.length === 0 ? (
          <div className="empty-state">
            <span className="empty-state-title">No assigned sessions</span>
            <p className="empty-state-description">
              Sessions assigned to you for reference report submission will appear here.
            </p>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Session Name</th>
                  <th>Reference Report</th>
                  <th>Report Status</th>
                  <th>Session Readiness</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {assignments.map(item => {
                  const isReadyForReflection =
                    item.session_readiness_status === 'ready_for_assessment' ||
                    item.session_readiness_status === 'ready';
                  const existingSub = submissions.find(s => s.lesson_id === item.lesson_id);

                  return (
                    <tr key={item.lesson_id}>
                      <td>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                          <button
                            type="button"
                            className="btn-link"
                            style={{
                              padding: 0,
                              background: 'none',
                              border: 'none',
                              textAlign: 'left',
                              cursor: 'pointer',
                              font: 'inherit',
                              color: 'inherit',
                            }}
                            onClick={() => {
                              setPreselectedLessonId(item.lesson_id);
                              setSelectedLessonId(item.lesson_id);
                              setView('submit-document');
                            }}
                          >
                            <span className="cell-lesson-title" style={{ color: 'var(--accent)', textDecoration: 'underline' }}>
                              {item.lesson_title}
                            </span>
                          </button>
                          <span style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>
                            {item.session_date || item.subject}
                          </span>
                        </div>
                      </td>
                      <td>
                        {item.document_uploaded ? (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                            <span style={{ fontSize: '13px', color: 'var(--text)' }}>
                              {item.report_title || item.original_filename}{' '}
                              {item.version ? `(v${item.version})` : ''}
                            </span>
                            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                              {item.entry_mode === 'text' || item.entry_mode === 'direct_text'
                                ? 'Written Report'
                                : 'Uploaded Document'}
                            </span>
                          </div>
                        ) : (
                          <span style={{ fontSize: '12.5px', color: 'var(--text-dim)' }}>Pending</span>
                        )}
                      </td>
                      <td>
                        <ContributorDocStatusBadge status={item.processing_status} />
                      </td>
                      <td>
                        <span
                          className={`badge ${
                            isReadyForReflection
                              ? 'badge-completed'
                              : item.session_readiness_status === 'extraction_failed'
                                ? 'badge-failed'
                                : 'badge-processing'
                          }`}
                        >
                          <span className="badge-dot" />
                          <span>{formatReadinessLabel(item.session_readiness_status)}</span>
                        </span>
                      </td>
                      <td>
                        <div className="cell-actions-group">
                          {item.processing_status === 'failed' && (
                            <button
                              className="btn btn-secondary btn-sm"
                              onClick={() => handleRetryDocument(item.lesson_id)}
                              disabled={retryingLessonId === item.lesson_id}
                            >
                              {retryingLessonId === item.lesson_id ? (
                                <Loader2 size={12} className="spin" />
                              ) : (
                                <>
                                  <RotateCcw size={12} />
                                  <span>Retry</span>
                                </>
                              )}
                            </button>
                          )}

                          <button
                            className="btn btn-secondary btn-sm"
                            onClick={() => {
                              setPreselectedLessonId(item.lesson_id);
                              setSelectedLessonId(item.lesson_id);
                              setView('submit-document');
                            }}
                          >
                            <FileUp size={12} />
                            <span>Open Session</span>
                          </button>

                          <button
                            className={`btn ${item.document_uploaded ? 'btn-secondary' : 'btn-primary'} btn-sm`}
                            onClick={() => {
                              setPreselectedLessonId(item.lesson_id);
                              setSelectedLessonId(item.lesson_id);
                              setEntryMode('file_upload');
                              setView('submit-document');
                            }}
                          >
                            <Upload size={12} />
                            <span>{item.document_uploaded ? 'Upload File' : 'Upload Document'}</span>
                          </button>

                          <button
                            className="btn btn-secondary btn-sm"
                            onClick={() => {
                              setPreselectedLessonId(item.lesson_id);
                              setSelectedLessonId(item.lesson_id);
                              setEntryMode('direct_text');
                              setView('submit-document');
                            }}
                          >
                            <PenLine size={12} />
                            <span>Write Report</span>
                          </button>

                          {existingSub ? (
                            <button
                              className="btn btn-secondary btn-sm"
                              onClick={() => onSelectSubmission(existingSub)}
                            >
                              <span>View Status</span>
                            </button>
                          ) : (
                            <button
                              className="btn btn-secondary btn-sm"
                              onClick={() => {
                                setPreselectedReflectionLessonId(item.lesson_id);
                                setView('write-reflection');
                              }}
                            >
                              <span>Write Reflection</span>
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* SESSIONS CREATED BY CONTRIBUTOR */}
      <div className="table-card" style={{ marginTop: '16px' }}>
        <div className="table-card-header">
          <span className="table-card-title">Sessions Created by You</span>
          <button className="btn btn-secondary btn-sm" onClick={() => setView('manage-sessions')}>
            <Plus size={13} />
            <span>Create / Manage Sessions</span>
          </button>
        </div>

        {managedLessons.length === 0 ? (
          <div className="empty-state">
            <span className="empty-state-title">No sessions created by you yet</span>
            <p className="empty-state-description">
              Contributors can create C4 assessment sessions and assign 3 contributors.
            </p>
            <button className="btn btn-primary btn-sm" onClick={() => setView('manage-sessions')}>
              <Plus size={13} />
              <span>Create Session</span>
            </button>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Session Name</th>
                  <th>Date</th>
                  <th>Readiness</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {managedLessons.map(lesson => {
                  const isReady =
                    lesson.readiness_status === 'ready_for_assessment' || lesson.readiness_status === 'ready';
                  return (
                    <tr key={lesson.id}>
                      <td>
                        <span className="cell-lesson-title">{lesson.title}</span>
                      </td>
                      <td>
                        <span className="cell-mono">{lesson.session_date || 'Current'}</span>
                      </td>
                      <td>
                        <span className={`badge ${isReady ? 'badge-completed' : 'badge-processing'}`}>
                          <span className="badge-dot" />
                          <span>{formatReadinessLabel(lesson.readiness_status)}</span>
                        </span>
                      </td>
                      <td>
                        <div className="cell-actions-group">
                          <button className="btn btn-secondary btn-sm" onClick={() => setView('manage-sessions')}>
                            View Status
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ASSESSMENT SESSIONS AVAILABLE FOR REFLECTION */}
      <div className="table-card" style={{ marginTop: '16px' }}>
        <div className="table-card-header">
          <span className="table-card-title">Assessment Sessions — Write Reflection</span>
        </div>

        {lessons.length === 0 ? (
          <div className="empty-state">
            <span className="empty-state-title">No sessions published yet</span>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Session Name</th>
                  <th>Date</th>
                  <th>Reflection Status</th>
                  <th>Rating</th>
                  <th style={{ textAlign: 'right' }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {lessons.map(lesson => {
                  const sub = submissions.find(s => s.lesson_id === lesson.id);
                  const isReady = lesson.is_ready_for_submission !== false;
                  return (
                    <tr key={lesson.id}>
                      <td>
                        <span className="cell-lesson-title">{lesson.title}</span>
                      </td>
                      <td>
                        <span className="cell-mono">{lesson.session_date || 'Current'}</span>
                      </td>
                      <td>
                        {sub ? (
                          <StatusBadge status={sub.assessment_status} />
                        ) : isReady ? (
                          <span className="badge badge-completed">
                            <span className="badge-dot" />
                            <span>Open for Reflection</span>
                          </span>
                        ) : (
                          <span className="badge badge-pending">
                            <span className="badge-dot" />
                            <span>{formatReadinessLabel(lesson.readiness_status)}</span>
                          </span>
                        )}
                      </td>
                      <td>
                        {sub && sub.rating !== null ? (
                          <span className="cell-rating-badge">{sub.rating} / 5</span>
                        ) : (
                          <span style={{ color: 'var(--text-dim)' }}>—</span>
                        )}
                      </td>
                      <td>
                        <div className="cell-actions-group">
                          {sub ? (
                            <button className="btn btn-secondary btn-sm" onClick={() => onSelectSubmission(sub)}>
                              View Submission
                            </button>
                          ) : (
                            <button
                              className="btn btn-primary btn-sm"
                              disabled={!isReady}
                              onClick={() => {
                                setPreselectedReflectionLessonId(lesson.id);
                                setView('write-reflection');
                              }}
                            >
                              Write Reflection
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}

// ============================================================================
// 6. STUDENT DASHBOARD — FOCUS ON REFLECTION
// ============================================================================
function StudentDashboardOverview({
  lessons,
  submissions,
  analytics,
  loading,
  onStartReflection,
  onSelectSubmission,
  onNavigate,
}: {
  lessons: Lesson[];
  submissions: Submission[];
  analytics: DashboardAnalyticsResponse | null;
  loading: boolean;
  onStartReflection: (lessonId: number) => void;
  onSelectSubmission: (sub: Submission) => void;
  onNavigate: (view: string) => void;
}) {
  const completed = submissions.filter(s => s.assessment_status === 'completed');
  const personalTrend = analytics?.personal_trend || [];
  const personalCompletion = analytics?.personal_completion || [];
  const PIE_COLORS = ['#6366f1', '#94a3b8'];

  return (
    <>
      <div className="page-header">
        <div className="page-header-text">
          <h1 className="page-title">Student Dashboard</h1>
          <p className="page-description">View available sessions, submit reflections, and check your results.</p>
        </div>
        <button className="btn btn-primary" onClick={() => onNavigate('submit')}>
          <Send size={14} />
          <span>Submit Reflection</span>
        </button>
      </div>

      <div className="kpi-grid">
        <div className="kpi-card">
          <span className="kpi-label">Available Sessions</span>
          <span className="kpi-value">{loading ? '—' : lessons.length}</span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Submitted Reflections</span>
          <span className="kpi-value">{loading ? '—' : submissions.length}</span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Assessment Results</span>
          <span className="kpi-value">{loading ? '—' : completed.length}</span>
        </div>
      </div>

      {/* PERSONAL PERFORMANCE CHARTS */}
      <div className="analytics-charts-grid" style={{ marginBottom: '16px' }}>
        <div className="chart-card">
          <div className="chart-card-header">
            <span className="chart-card-title">Personal Rating Trend</span>
          </div>
          {personalTrend.length === 0 ? (
            <div className="chart-empty">Submit reflections to view your personal rating trend across sessions.</div>
          ) : (
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={personalTrend} margin={{ top: 8, right: 12, left: -16, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="session_title" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 5]} ticks={[1, 2, 3, 4, 5]} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="rating"
                    name="Rating (1-5)"
                    stroke="#6366f1"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <div className="chart-card">
          <div className="chart-card-header">
            <span className="chart-card-title">Submission Completion</span>
          </div>
          {lessons.length === 0 ? (
            <div className="chart-empty">No sessions published yet.</div>
          ) : (
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={personalCompletion}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={45}
                    outerRadius={70}
                    paddingAngle={2}
                  >
                    {personalCompletion.map((_, idx) => (
                      <Cell key={idx} fill={PIE_COLORS[idx % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      {/* AVAILABLE SESSIONS TABLE */}
      <div className="table-card">
        <div className="table-card-header">
          <span className="table-card-title">Available Sessions</span>
        </div>

        {lessons.length === 0 ? (
          <div className="empty-state">
            <span className="empty-state-title">No sessions available yet</span>
            <p className="empty-state-description">Published Business Forecasting sessions will appear here.</p>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Session</th>
                  <th>Date</th>
                  <th>Submission Status</th>
                  <th>Result</th>
                  <th style={{ textAlign: 'right' }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {lessons.map(lesson => {
                  const sub = submissions.find(s => s.lesson_id === lesson.id);
                  const isReady = lesson.is_ready_for_submission !== false;
                  return (
                    <tr key={lesson.id}>
                      <td>
                        <span className="cell-lesson-title">{lesson.title}</span>
                      </td>
                      <td>
                        <span className="cell-mono">{lesson.session_date || 'Current'}</span>
                      </td>
                      <td>
                        {sub ? (
                          <StatusBadge status={sub.assessment_status} />
                        ) : isReady ? (
                          <span className="badge badge-completed">
                            <span className="badge-dot" />
                            <span>Open</span>
                          </span>
                        ) : (
                          <span className="badge badge-pending">
                            <span className="badge-dot" />
                            <span>{formatReadinessLabel(lesson.readiness_status)}</span>
                          </span>
                        )}
                      </td>
                      <td>
                        {sub && sub.rating !== null ? (
                          <span className="cell-rating-badge">{sub.rating} / 5</span>
                        ) : (
                          <span style={{ color: 'var(--text-dim)' }}>—</span>
                        )}
                      </td>
                      <td>
                        <div className="cell-actions-group">
                          {sub ? (
                            <button className="btn btn-secondary btn-sm" onClick={() => onSelectSubmission(sub)}>
                              View Result
                            </button>
                          ) : (
                            <button
                              className="btn btn-primary btn-sm"
                              disabled={!isReady}
                              onClick={() => onStartReflection(lesson.id)}
                            >
                              Reflect
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}

function StudentLessonsView({
  lessons,
  submissions,
  onStartReflection,
  onViewSubmission,
}: {
  lessons: Lesson[];
  submissions: Submission[];
  loading?: boolean;
  onStartReflection: (lessonId: number) => void;
  onViewSubmission: (sub: Submission) => void;
}) {
  return (
    <>
      <div className="page-header">
        <div className="page-header-text">
          <h1 className="page-title">Available Sessions</h1>
          <p className="page-description">Select a session to submit your 500-character reflection.</p>
        </div>
      </div>

      {lessons.length === 0 ? (
        <div className="table-card">
          <div className="empty-state">
            <span className="empty-state-title">No sessions published yet</span>
          </div>
        </div>
      ) : (
        <div className="lessons-grid">
          {lessons.map(lesson => {
            const sub = submissions.find(s => s.lesson_id === lesson.id);
            const isCompleted = sub?.assessment_status === 'completed';
            const isReady = lesson.is_ready_for_submission !== false;

            return (
              <div className="lesson-card" key={lesson.id}>
                <div className="lesson-card-header">
                  <span className="lesson-card-meta">{lesson.session_date || SUBJECT}</span>
                  {sub ? (
                    <StatusBadge status={sub.assessment_status} />
                  ) : isReady ? (
                    <span className="badge badge-completed">
                      <span className="badge-dot" />
                      <span>Ready</span>
                    </span>
                  ) : (
                    <span className="badge badge-pending">
                      <span className="badge-dot" />
                      <span>{formatReadinessLabel(lesson.readiness_status)}</span>
                    </span>
                  )}
                </div>

                <h3 className="lesson-card-title">{lesson.title}</h3>
                <p className="lesson-card-body">
                  {isReady
                    ? 'Write a concise reflection from memory summarizing key concepts from this session.'
                    : 'This session is currently preparing reference documents.'}
                </p>

                <div className="lesson-card-footer">
                  {sub ? (
                    <button className="btn btn-secondary btn-sm" onClick={() => onViewSubmission(sub)}>
                      <span>{isCompleted ? `View Result (${sub.rating}/5)` : 'View Submission'}</span>
                    </button>
                  ) : (
                    <button
                      className="btn btn-primary btn-sm"
                      disabled={!isReady}
                      onClick={() => onStartReflection(lesson.id)}
                    >
                      <span>Start Reflection</span>
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </>
  );
}

// ============================================================================
// STUDENT & CONTRIBUTOR REFLECTION SUBMISSION VIEW
// ============================================================================
function ReflectionSubmitView({
  lessons,
  submissions,
  initialLessonId,
  request,
  onSelectSubmission,
  onDownload,
  onDone,
  onError,
}: {
  lessons: Lesson[];
  submissions: Submission[];
  initialLessonId: number | null;
  request: (path: string, options?: RequestInit) => Promise<any>;
  onSelectSubmission?: (sub: Submission) => void;
  onDownload?: (path: string, name: string) => Promise<void>;
  onDone: () => void | Promise<void>;
  onError: (msg: string) => void;
}) {
  const readyLessons = useMemo(
    () => lessons.filter(l => l.is_ready_for_submission !== false),
    [lessons]
  );

  const [selectedLessonId, setSelectedLessonId] = useState<string>(() => {
    if (initialLessonId && lessons.some(l => l.id === initialLessonId)) {
      return initialLessonId.toString();
    }
    return readyLessons[0]?.id?.toString() || lessons[0]?.id?.toString() || '';
  });
  const [text, setText] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState('');
  const [submitSuccess, setSubmitSuccess] = useState('');

  useEffect(() => {
    if (initialLessonId && lessons.some(l => l.id === initialLessonId)) {
      setSelectedLessonId(initialLessonId.toString());
    } else if (!selectedLessonId && (readyLessons[0] || lessons[0])) {
      setSelectedLessonId((readyLessons[0]?.id || lessons[0]?.id).toString());
    }
  }, [initialLessonId, lessons]);

  const selectedLesson = lessons.find(l => l.id.toString() === selectedLessonId);
  const existingSubmission = submissions.find(s => s.lesson_id.toString() === selectedLessonId);
  const isLessonReady = selectedLesson ? selectedLesson.is_ready_for_submission !== false : false;

  const charCount = text.length;
  const maxChars = 500;
  const isWarning = charCount >= 450 && charCount <= maxChars;
  const isOver = charCount > maxChars;
  const isEmpty = text.trim().length === 0;
  const isValid = !isEmpty && !isOver && selectedLessonId !== '' && isLessonReady && !existingSubmission;

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!isValid || submitting) return;

    setSubmitting(true);
    setSubmitError('');
    setSubmitSuccess('');
    try {
      await request(`/lessons/${selectedLessonId}/submissions`, {
        method: 'POST',
        body: JSON.stringify({ text }),
      });
      setText('');
      setSubmitSuccess('Your reflection has been submitted and saved for assessment.');
      await onDone();
    } catch (err: any) {
      // Preserve `text` if submission or assessment fails
      const errMsg = err.message || 'Failed to submit reflection.';
      setSubmitError(errMsg);
      onError(errMsg);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="reflection-container">
      <div className="page-header">
        <div className="page-header-text">
          <h1 className="page-title">Write Reflection</h1>
          <p className="page-description">
            Select an authorized Business Forecasting session and write a concise reflection from memory (up to 500
            characters, including spaces and punctuation).
          </p>
        </div>
      </div>

      <form className="editor-card" onSubmit={handleSubmit}>
        <div className="form-group">
          <label className="form-label" htmlFor="reflection-session-select">
            Assessment Session
          </label>
          <select
            id="reflection-session-select"
            value={selectedLessonId}
            onChange={e => {
              setSelectedLessonId(e.target.value);
              setSubmitError('');
              setSubmitSuccess('');
            }}
            disabled={submitting}
          >
            <option value="">-- Select a session --</option>
            {lessons.map(l => {
              const sub = submissions.find(s => s.lesson_id === l.id);
              const ready = l.is_ready_for_submission !== false;
              return (
                <option key={l.id} value={l.id}>
                  {l.title}
                  {sub ? ' (Submitted)' : !ready ? ` (${formatReadinessLabel(l.readiness_status)})` : ' (Ready)'}
                </option>
              );
            })}
          </select>
        </div>

        {/* SESSION TITLE & REFLECTION INSTRUCTIONS */}
        {selectedLesson && (
          <div
            style={{
              padding: '10px 14px',
              backgroundColor: 'var(--bg-subtle)',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border)',
              display: 'flex',
              flexDirection: 'column',
              gap: '4px',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--text)' }}>
                {selectedLesson.title}
              </span>
              <span className="badge badge-draft">
                {selectedLesson.session_date || selectedLesson.subject}
              </span>
            </div>
            <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', margin: 0 }}>
              <strong>Instructions:</strong> Summarize the core forecasting concepts, methods, and key takeaways from
              this session in your own words without consulting external notes. Maximum length: 500 characters
              (including spaces and punctuation).
            </p>
          </div>
        )}

        {/* SUBMISSION OR ASSESSMENT STATUS ALERTS */}
        {selectedLesson && !isLessonReady && (
          <div className="form-error-alert" role="alert">
            <Clock size={15} />
            <span>
              Status: <strong>{formatReadinessLabel(selectedLesson.readiness_status)}</strong>. Reflections open
              automatically once all 3 reference documents are processed.
            </span>
          </div>
        )}

        {submitError && (
          <div className="form-error-alert" role="alert">
            <AlertCircle size={15} />
            <span>{submitError}</span>
          </div>
        )}

        {submitSuccess && (
          <div className="form-success-alert" role="status">
            <CheckCircle2 size={15} />
            <span>{submitSuccess}</span>
          </div>
        )}

        {existingSubmission && (
          <div className="form-success-alert" role="status">
            <CheckCircle2 size={15} />
            <span>
              You have already submitted a reflection for this session. Duplicate submissions are disabled.
            </span>
          </div>
        )}

        <div className="form-group">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <label className="form-label" htmlFor="reflection-textarea">
              Your Reflection
            </label>
            <span className={`char-counter-text ${isOver ? 'danger' : isWarning ? 'warning' : ''}`}>
              {charCount} / {maxChars}
            </span>
          </div>

          <textarea
            id="reflection-textarea"
            className="reflection-textarea"
            required
            placeholder="Write what you recall from this session (up to 500 characters)..."
            value={text}
            onChange={e => {
              setText(e.target.value);
              if (submitError) setSubmitError('');
            }}
            disabled={submitting || !isLessonReady || Boolean(existingSubmission)}
          />

          <div className="char-meter">
            <div
              className="char-meter-fill"
              style={{
                width: `${Math.min(100, (charCount / maxChars) * 100)}%`,
                background: isOver ? 'var(--danger)' : isWarning ? 'var(--warning)' : 'var(--accent)',
              }}
            />
          </div>

          {isOver && (
            <div className="char-counter-bar">
              <span style={{ color: 'var(--danger)', fontWeight: 500 }}>
                Exceeds the 500-character limit by {charCount - maxChars} character(s). Please shorten your reflection
                to 500 characters or fewer.
              </span>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button type="submit" className="btn btn-primary" disabled={!isValid || submitting}>
            {submitting ? (
              <>
                <Loader2 size={15} className="spin" />
                <span>Submitting...</span>
              </>
            ) : (
              <>
                <Send size={14} />
                <span>Submit Reflection</span>
              </>
            )}
          </button>
        </div>
      </form>

      {/* SAVED SUBMISSION & ASSESSMENT STATUS CARD */}
      {existingSubmission && (
        <div className="editor-card" style={{ marginTop: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px' }}>
            <div>
              <h3 style={{ fontSize: '14.5px', fontWeight: 600, color: 'var(--text)', margin: 0 }}>
                Saved Reflection & Assessment Status
              </h3>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                Submitted {new Date(existingSubmission.submitted_at).toLocaleString()} ·{' '}
                {existingSubmission.character_count} / 500 characters
              </span>
            </div>
            <StatusBadge status={existingSubmission.assessment_status} />
          </div>

          <div className="reflection-quote-box">{existingSubmission.text}</div>

          {existingSubmission.rating !== null && (
            <div className="score-summary-grid">
              <div className="score-box">
                <span className="score-box-label">Final Rating</span>
                <span className="score-box-value">{existingSubmission.rating} / 5</span>
              </div>
              <div className="score-box">
                <span className="score-box-label">
                  {existingSubmission.assessment_mode === 'c4' ? 'Matched Concepts' : 'Recall Score'}
                </span>
                <span className="score-box-value">
                  {existingSubmission.assessment_mode === 'c4' && existingSubmission.matched_concepts_count !== null
                    ? `${existingSubmission.matched_concepts_count} / ${existingSubmission.total_pool_concepts ?? 15}`
                    : `${existingSubmission.recall_percentage ?? 0}%`}
                </span>
              </div>
              <div className="score-box">
                <span className="score-box-label">Coverage</span>
                <span className="score-box-value">{existingSubmission.recall_percentage ?? 0}%</span>
              </div>
            </div>
          )}

          {existingSubmission.feedback && (
            <div className="feedback-box">
              <p>{existingSubmission.feedback}</p>
            </div>
          )}

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
            {onSelectSubmission && (
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => onSelectSubmission(existingSubmission)}
              >
                <span>View Details</span>
              </button>
            )}
            {onDownload && existingSubmission.assessment_status === 'completed' && (
              <button
                type="button"
                className="btn btn-primary btn-sm"
                onClick={() =>
                  onDownload(
                    `/submissions/${existingSubmission.id}/report.pdf`,
                    `recallai-report-${existingSubmission.id}.pdf`
                  )
                }
              >
                <Download size={13} />
                <span>Download PDF</span>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// SUBMISSIONS LIST VIEW (FACULTY & STUDENT)
// ============================================================================
function SubmissionsListView({
  faculty,
  submissions,
  request,
  onRefresh,
  onSelectSubmission,
  onDownload,
  onError,
}: {
  faculty: boolean;
  submissions: Submission[];
  loading?: boolean;
  request: (path: string, options?: RequestInit) => Promise<any>;
  onRefresh: () => Promise<void>;
  onSelectSubmission: (sub: Submission) => void;
  onDownload: (path: string, name: string) => Promise<void>;
  onError: (msg: string) => void;
}) {
  const [search, setSearch] = useState('');
  const [topicFilter, setTopicFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [sortKey, setSortKey] = useState<string>('submitted_at');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [retryingId, setRetryingId] = useState<number | null>(null);

  const availableLessons = useMemo(() => {
    const map = new Map<number, string>();
    submissions.forEach(s => map.set(s.lesson_id, s.lesson));
    return Array.from(map.entries()).map(([id, title]) => ({ id, title }));
  }, [submissions]);

  const filtered = useMemo(() => {
    let result = [...submissions];

    if (search.trim()) {
      const q = search.toLowerCase();
      if (faculty) {
        result = result.filter(
          s =>
            s.student_name?.toLowerCase().includes(q) ||
            s.student_email?.toLowerCase().includes(q) ||
            s.lesson?.toLowerCase().includes(q)
        );
      } else {
        result = result.filter(s => s.lesson?.toLowerCase().includes(q));
      }
    }

    if (topicFilter !== 'all') {
      result = result.filter(s => s.lesson_id.toString() === topicFilter);
    }

    if (statusFilter !== 'all') {
      result = result.filter(s => s.assessment_status === statusFilter);
    }

    result.sort((a: any, b: any) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;
      const cmp = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      return sortDirection === 'asc' ? cmp : -cmp;
    });

    return result;
  }, [submissions, search, topicFilter, statusFilter, sortKey, sortDirection, faculty]);

  function handleSort(key: string) {
    if (sortKey === key) {
      setSortDirection(prev => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDirection('desc');
    }
  }

  async function handleRetry(id: number) {
    setRetryingId(id);
    try {
      await request(`/submissions/${id}/retry`, { method: 'POST' });
      await onRefresh();
    } catch (err: any) {
      onError(err.message || 'Retry failed');
    } finally {
      setRetryingId(null);
    }
  }

  return (
    <>
      <div className="page-header">
        <div className="page-header-text">
          <h1 className="page-title">{faculty ? 'Student Submissions' : 'My Submissions'}</h1>
          <p className="page-description">
            {faculty
              ? 'Review student reflections, assessment statuses, and scores.'
              : 'Track your submitted reflections and assessment results.'}
          </p>
        </div>
      </div>

      <div className="filter-card">
        <div className="filter-grid">
          <div className="filter-item">
            <label className="filter-label">{faculty ? 'Search Student or Session' : 'Search Session'}</label>
            <div className="input-with-icon">
              <Search size={14} />
              <input
                type="text"
                placeholder={faculty ? 'Name, email, or session...' : 'Session title...'}
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
            </div>
          </div>

          <div className="filter-item">
            <label className="filter-label">Session</label>
            <select value={topicFilter} onChange={e => setTopicFilter(e.target.value)}>
              <option value="all">All Sessions</option>
              {availableLessons.map(l => (
                <option key={l.id} value={l.id}>
                  {l.title}
                </option>
              ))}
            </select>
          </div>

          <div className="filter-item">
            <label className="filter-label">Status</label>
            <select value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
              <option value="all">All Statuses</option>
              <option value="completed">Completed</option>
              <option value="processing">Processing</option>
              <option value="pending">Pending</option>
              <option value="failed">Failed</option>
            </select>
          </div>

          <div className="filter-item">
            <label className="filter-label">Sort By</label>
            <select
              value={`${sortKey}:${sortDirection}`}
              onChange={e => {
                const [k, d] = e.target.value.split(':');
                setSortKey(k);
                setSortDirection(d as 'asc' | 'desc');
              }}
            >
              <option value="submitted_at:desc">Newest First</option>
              <option value="submitted_at:asc">Oldest First</option>
              <option value="rating:desc">Rating (High to Low)</option>
              <option value="recall_percentage:desc">Recall % (High to Low)</option>
            </select>
          </div>

          {(search || topicFilter !== 'all' || statusFilter !== 'all') && (
            <button
              className="btn btn-ghost btn-sm"
              onClick={() => {
                setSearch('');
                setTopicFilter('all');
                setStatusFilter('all');
              }}
            >
              Clear
            </button>
          )}
        </div>
      </div>

      <div className="table-card">
        {filtered.length === 0 ? (
          <div className="empty-state">
            <span className="empty-state-title">No matching submissions</span>
            <p className="empty-state-description">No submissions match the selected filters.</p>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  {faculty && (
                    <th className="sortable" onClick={() => handleSort('student_name')}>
                      Student
                    </th>
                  )}
                  <th className="sortable" onClick={() => handleSort('lesson')}>
                    Session
                  </th>
                  <th className="sortable" onClick={() => handleSort('submitted_at')}>
                    Submitted
                  </th>
                  <th className="sortable" onClick={() => handleSort('character_count')}>
                    Length
                  </th>
                  <th className="sortable" onClick={() => handleSort('assessment_status')}>
                    Status
                  </th>
                  <th className="sortable" onClick={() => handleSort('recall_percentage')}>
                    Score
                  </th>
                  <th className="sortable" onClick={() => handleSort('rating')}>
                    Rating
                  </th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(item => (
                  <tr key={item.id}>
                    {faculty && (
                      <td>
                        <div className="cell-student-info">
                          <div className="cell-avatar">{item.student_name?.slice(0, 1).toUpperCase() || '?'}</div>
                          <div className="cell-student-text">
                            <span className="cell-student-name">{item.student_name || 'Student'}</span>
                            <span className="cell-student-email">{item.student_email}</span>
                          </div>
                        </div>
                      </td>
                    )}

                    <td>
                      <span className="cell-lesson-title">{item.lesson}</span>
                    </td>

                    <td>
                      <span className="cell-mono">{new Date(item.submitted_at).toLocaleDateString()}</span>
                    </td>

                    <td>
                      <span className="cell-mono">{item.character_count} / 500</span>
                    </td>

                    <td>
                      <StatusBadge status={item.assessment_status} />
                    </td>

                    <td>
                      <span className="cell-score">
                        {item.assessment_mode === 'c4' && item.matched_concepts_count != null
                          ? `${item.matched_concepts_count}/${item.total_pool_concepts ?? 15} (${item.recall_percentage ?? 0}%)`
                          : item.recall_percentage !== null
                            ? `${item.recall_percentage}%`
                            : '—'}
                      </span>
                    </td>

                    <td>
                      {item.rating !== null ? (
                        <div className="cell-rating-badge">
                          <Star size={12} fill="currentColor" />
                          <span>{item.rating}/5</span>
                        </div>
                      ) : (
                        <span style={{ color: 'var(--text-dim)' }}>—</span>
                      )}
                    </td>

                    <td>
                      <div className="cell-actions-group">
                        <button
                          className="btn btn-secondary btn-sm"
                          onClick={() => onSelectSubmission(item)}
                        >
                          View
                        </button>

                        {faculty && item.assessment_status === 'failed' && (
                          <button
                            className="btn btn-secondary btn-sm"
                            onClick={() => handleRetry(item.id)}
                            disabled={retryingId === item.id}
                          >
                            {retryingId === item.id ? <Loader2 size={12} className="spin" /> : 'Retry'}
                          </button>
                        )}

                        {item.assessment_status === 'completed' && (
                          <button
                            className="btn-icon"
                            onClick={() =>
                              onDownload(`/submissions/${item.id}/report.pdf`, `recallai-report-${item.id}.pdf`)
                            }
                            title="Download PDF Report"
                          >
                            <Download size={14} />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}

// ============================================================================
// STUDENT RESULTS VIEW
// ============================================================================
function MyResultsView({
  submissions,
  onSelectSubmission,
  onDownload,
}: {
  submissions: Submission[];
  onSelectSubmission: (sub: Submission) => void;
  onDownload: (path: string, name: string) => Promise<void>;
}) {
  return (
    <>
      <div className="page-header">
        <div className="page-header-text">
          <h1 className="page-title">Assessment Results</h1>
          <p className="page-description">View your completed assessment scores and download PDF reports.</p>
        </div>
      </div>

      <div className="table-card">
        {submissions.length === 0 ? (
          <div className="empty-state">
            <span className="empty-state-title">No completed results yet</span>
            <p className="empty-state-description">Completed assessments will appear here once evaluated.</p>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Session</th>
                  <th>Submitted</th>
                  <th>Score</th>
                  <th>Rating</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {submissions.map(item => (
                  <tr key={item.id}>
                    <td>
                      <span className="cell-lesson-title">{item.lesson}</span>
                    </td>
                    <td>
                      <span className="cell-mono">{new Date(item.submitted_at).toLocaleDateString()}</span>
                    </td>
                    <td>
                      <span className="cell-score">
                        {item.assessment_mode === 'c4' && item.matched_concepts_count != null
                          ? `${item.matched_concepts_count}/${item.total_pool_concepts ?? 15} (${item.recall_percentage}%)`
                          : `${item.recall_percentage}%`}
                      </span>
                    </td>
                    <td>
                      <span className="cell-rating-badge">{item.rating} / 5</span>
                    </td>
                    <td>
                      <div className="cell-actions-group">
                        <button className="btn btn-secondary btn-sm" onClick={() => onSelectSubmission(item)}>
                          Details
                        </button>
                        <button
                          className="btn btn-secondary btn-sm"
                          onClick={() =>
                            onDownload(`/submissions/${item.id}/report.pdf`, `recallai-report-${item.id}.pdf`)
                          }
                        >
                          <Download size={13} />
                          <span>PDF</span>
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}

// ============================================================================
// REPORTS VIEW
// ============================================================================
function ReportsView({
  onDownload,
  submissionsCount,
}: {
  onDownload: (path: string, name: string) => Promise<void>;
  submissionsCount: number;
}) {
  return (
    <>
      <div className="page-header">
        <div className="page-header-text">
          <h1 className="page-title">Reports</h1>
          <p className="page-description">Export class assessment results or download individual PDF reports.</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px' }}>
        <div className="editor-card">
          <h3 style={{ fontSize: '15px', color: 'var(--text)' }}>Class Results Export (.XLSX)</h3>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Download a spreadsheet of all student submissions, character counts, concept match scores, and ratings.
          </p>
          <div>
            <button
              className="btn btn-primary"
              onClick={() => onDownload('/faculty/export.xlsx', 'recallai-results.xlsx')}
            >
              <Download size={14} />
              <span>Download Excel ({submissionsCount} records)</span>
            </button>
          </div>
        </div>

        <div className="editor-card">
          <h3 style={{ fontSize: '15px', color: 'var(--text)' }}>Individual PDF Reports</h3>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Individual PDF reports can be downloaded from the Submissions table or when viewing a submission.
          </p>
        </div>
      </div>
    </>
  );
}

// ============================================================================
// SUBMISSION DETAIL SLIDE-OVER DRAWER
// ============================================================================
function SubmissionDetailDrawer({
  submission,
  faculty,
  onClose,
  onDownload,
  request,
  onRefresh,
  onError,
}: {
  submission: Submission;
  faculty: boolean;
  onClose: () => void;
  onDownload: (path: string, name: string) => Promise<void>;
  request: (path: string, options?: RequestInit) => Promise<any>;
  onRefresh: () => Promise<void>;
  onError: (msg: string) => void;
}) {
  const [retrying, setRetrying] = useState(false);

  const details = submission.assessment_details || {};
  const conceptResults: ConceptResult[] = faculty ? details.concept_results || [] : [];
  const missingConcepts: string[] = faculty ? details.missing_concepts || [] : [];
  const misconceptions: string[] = faculty ? details.misconceptions || [] : [];
  const strengths: string[] = details.strengths || [];
  const areasForImprovement: string[] = details.areas_for_improvement || [];
  const isCompleted = submission.assessment_status === 'completed';
  const isFailed = submission.assessment_status === 'failed';
  const isC4 = submission.assessment_mode === 'c4' || details.assessment_mode === 'c4';

  const matchedCount = submission.matched_concepts_count ?? details.matched_count ?? null;
  const totalPoolCount = submission.total_pool_concepts ?? details.total_distinct_concepts ?? 15;

  async function handleAssess() {
    setRetrying(true);
    try {
      await request(`/submissions/${submission.id}/assess`, { method: 'POST' });
      await onRefresh();
    } catch (err: any) {
      onError(err.message || 'Assessment request failed');
    } finally {
      setRetrying(false);
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="drawer-panel" onClick={e => e.stopPropagation()}>
        <div className="drawer-header">
          <div>
            <span className="drawer-kicker">{submission.lesson}</span>
            <h2 className="drawer-title">Submission #{submission.id}</h2>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              {new Date(submission.submitted_at).toLocaleString()}
            </span>
          </div>
          <button className="btn-icon" onClick={onClose} aria-label="Close panel">
            <X size={16} />
          </button>
        </div>

        <div className="drawer-body">
          <div className="drawer-section">
            <div
              className="cell-student-info"
              style={{
                padding: '10px 12px',
                backgroundColor: 'var(--bg-subtle)',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border)',
              }}
            >
              <div className="cell-avatar">{submission.student_name?.slice(0, 1).toUpperCase() || '?'}</div>
              <div className="cell-student-text">
                <span className="cell-student-name">{submission.student_name || 'Student'}</span>
                <span className="cell-student-email">{submission.student_email}</span>
              </div>
              <div style={{ marginLeft: 'auto' }}>
                <StatusBadge status={submission.assessment_status} />
              </div>
            </div>
          </div>

          {isCompleted && (
            <div className="drawer-section">
              <span className="drawer-section-title">Assessment Summary</span>
              <div className="score-summary-grid">
                <div className="score-box">
                  <span className="score-box-label">Rating</span>
                  <span className="score-box-value">{submission.rating} / 5</span>
                </div>
                <div className="score-box">
                  <span className="score-box-label">{isC4 ? 'Matched Concepts' : 'Recall'}</span>
                  <span className="score-box-value">
                    {isC4 && matchedCount !== null
                      ? `${matchedCount} / ${totalPoolCount}`
                      : `${submission.recall_percentage}%`}
                  </span>
                </div>
                <div className="score-box">
                  <span className="score-box-label">Coverage</span>
                  <span className="score-box-value">{submission.recall_percentage}%</span>
                </div>
              </div>
            </div>
          )}

          <div className="drawer-section">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="drawer-section-title">Reflection</span>
              <span style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>
                {submission.character_count} / 500 characters
              </span>
            </div>
            <div className="reflection-quote-box">{submission.text}</div>
          </div>

          {submission.feedback && (
            <div className="drawer-section">
              <span className="drawer-section-title">Constructive Feedback</span>
              <div className="feedback-box">
                <p>{submission.feedback}</p>
              </div>
            </div>
          )}

          {isCompleted && strengths.length > 0 && (
            <div className="drawer-section">
              <span className="drawer-section-title">Learning Strengths</span>
              <ul className="learning-list">
                {strengths.map((s, idx) => (
                  <li key={idx}>{s}</li>
                ))}
              </ul>
            </div>
          )}

          {isCompleted && areasForImprovement.length > 0 && (
            <div className="drawer-section">
              <span className="drawer-section-title">Areas for Improvement</span>
              <ul className="learning-list">
                {areasForImprovement.map((a, idx) => (
                  <li key={idx}>{a}</li>
                ))}
              </ul>
            </div>
          )}

          {/* FACULTY-ONLY CONCEPT BREAKDOWN */}
          {faculty && isCompleted && conceptResults.length > 0 && (
            <div className="drawer-section">
              <span className="drawer-section-title">Concept Breakdown (Faculty Only)</span>
              <div className="concept-grid">
                {conceptResults.map((cr, idx) => (
                  <div className="concept-card" key={idx}>
                    <div className="concept-card-top">
                      <span className="concept-name">{cr.concept}</span>
                      <ConceptStatusBadge status={cr.status} />
                    </div>
                    {cr.evidence && <div className="concept-evidence">“{cr.evidence}”</div>}
                    {cr.explanation && <span className="concept-explanation">{cr.explanation}</span>}
                  </div>
                ))}
              </div>
            </div>
          )}

          {faculty && isCompleted && (missingConcepts.length > 0 || misconceptions.length > 0) && (
            <div className="drawer-section">
              <span className="drawer-section-title">Unmatched / Misconceptions (Faculty Only)</span>
              {missingConcepts.length > 0 && (
                <div className="concept-tag-list">
                  {missingConcepts.map((c, i) => (
                    <span key={i} className="concept-pill">
                      {c}
                    </span>
                  ))}
                </div>
              )}
              {misconceptions.length > 0 && (
                <div className="concept-tag-list" style={{ marginTop: '4px' }}>
                  {misconceptions.map((m, i) => (
                    <span
                      key={i}
                      className="concept-pill"
                      style={{ color: 'var(--danger)', borderColor: 'var(--danger-border)' }}
                    >
                      {m}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {isFailed && (
            <div className="form-error-alert" role="alert">
              <AlertCircle size={15} />
              <span>
                {submission.feedback || 'Assessment failed. The submitted reflection has been safely saved.'}
              </span>
            </div>
          )}
        </div>

        <div className="drawer-footer">
          {faculty && (
            <button className="btn btn-secondary btn-sm" onClick={handleAssess} disabled={retrying}>
              {retrying ? (
                <>
                  <Loader2 size={13} className="spin" />
                  <span>Assessing...</span>
                </>
              ) : (
                <>
                  <RotateCcw size={13} />
                  <span>{isCompleted ? 'Re-assess' : 'Retry Assessment'}</span>
                </>
              )}
            </button>
          )}

          {isCompleted && (
            <button
              className="btn btn-primary btn-sm"
              onClick={() =>
                onDownload(`/submissions/${submission.id}/report.pdf`, `recallai-report-${submission.id}.pdf`)
              }
            >
              <FileText size={13} />
              <span>Download PDF</span>
            </button>
          )}

          <button className="btn btn-ghost btn-sm" style={{ marginLeft: 'auto' }} onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// STATUS BADGES
// ============================================================================
function StatusBadge({ status }: { status: string }) {
  switch (status) {
    case 'completed':
      return (
        <span className="badge badge-completed">
          <span className="badge-dot" />
          <span>Completed</span>
        </span>
      );
    case 'processing':
    case 'pending':
      return (
        <span className="badge badge-processing">
          <span className="badge-dot" />
          <span>Processing</span>
        </span>
      );
    case 'failed':
      return (
        <span className="badge badge-failed">
          <span className="badge-dot" />
          <span>Failed</span>
        </span>
      );
    default:
      return (
        <span className="badge badge-pending">
          <span className="badge-dot" />
          <span>Received</span>
        </span>
      );
  }
}

function ContributorDocStatusBadge({ status }: { status: string }) {
  switch (status) {
    case 'completed':
      return (
        <span className="badge badge-completed">
          <span className="badge-dot" />
          <span>Processed</span>
        </span>
      );
    case 'uploading':
    case 'processing':
      return (
        <span className="badge badge-processing">
          <span className="badge-dot" />
          <span>Processing</span>
        </span>
      );
    case 'uploaded':
      return (
        <span className="badge badge-processing">
          <span className="badge-dot" />
          <span>Submitted</span>
        </span>
      );
    case 'failed':
      return (
        <span className="badge badge-failed">
          <span className="badge-dot" />
          <span>Failed</span>
        </span>
      );
    default:
      return (
        <span className="badge badge-pending">
          <span className="badge-dot" />
          <span>Pending</span>
        </span>
      );
  }
}

function ConceptStatusBadge({ status }: { status: string }) {
  switch (status) {
    case 'recalled':
      return <span className="badge badge-completed">Recalled</span>;
    case 'partial':
      return <span className="badge badge-processing">Partial</span>;
    case 'misconception':
      return <span className="badge badge-failed">Misconception</span>;
    default:
      return <span className="badge badge-draft">Missing</span>;
  }
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
