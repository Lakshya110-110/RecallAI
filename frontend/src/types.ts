export type Role = 'student' | 'faculty' | 'contributor';

export interface User {
  id: number;
  email: string;
  name: string;
  role: Role;
  login_portal?: Role;
  is_contributor?: boolean;
  effective_roles?: string[];
  contributor_lesson_ids?: number[];
}

export interface StudentOption {
  id: number;
  name: string;
  email: string;
  role: string;
}

export interface Lesson {
  id: number;
  title: string;
  description?: string;
  subject: string;
  group_name: string;
  session_date: string;
  reference_text: string;
  concepts: string[];
  publication_status: string;
  assessment_mode?: 'legacy' | 'c4' | string;
  required_contributors?: number;
  readiness_status?:
    | 'ready'
    | 'awaiting_contributors'
    | 'awaiting_uploads'
    | 'processing_documents'
    | 'extraction_failed'
    | 'ready_for_assessment'
    | string;
  is_ready_for_submission?: boolean;
  is_authorized_contributor?: boolean;
  is_creator?: boolean;
  can_manage?: boolean;
  created_by?: number;
  creator_name?: string;
  creator_role?: string;
  assigned_contributors_count?: number;
  uploaded_documents_count?: number;
  processed_documents_count?: number;
  pending_contributors_count?: number;
  contributor_ids?: number[];
  submission_count?: number;
  completed_submission_count?: number;
  average_rating?: number | null;
  pool_ready?: boolean;
  distinct_concepts_count?: number;
  pool_warning?: string | null;
  created_at: string | null;
}

export interface ContributorStatusItem {
  student_id: number;
  student_name: string;
  student_email: string;
  assigned_at: string;
  document_uploaded: boolean;
  document_id: number | null;
  entry_mode?: 'file' | 'text' | string | null;
  report_title?: string | null;
  original_filename: string | null;
  processing_status: 'awaiting_upload' | 'uploaded' | 'processing' | 'completed' | 'failed' | string;
  error_message: string | null;
  extracted_keywords_count: number;
  extracted_keywords: string[];
}

export interface LessonContributorsResponse {
  lesson_id: number;
  assessment_mode: string;
  required_contributors: number;
  readiness_status: string;
  contributors: ContributorStatusItem[];
}

export interface DistinctPoolConcept {
  normalized_concept: string;
  display_concept: string;
  surface_forms: string[];
  source_document_ids: number[];
  source_contributor_ids: number[];
  occurrence_count: number;
}

export interface PoolSourceEntry {
  document_id: number;
  contributor_id: number;
  position: number;
  raw_keyword: string;
  normalized_keyword: string;
}

export interface LessonKeywordPoolResponse {
  lesson_id: number;
  readiness_status: string;
  pool_ready: boolean;
  pool_id?: number;
  version: number | null;
  total_raw_keywords: number;
  distinct_concepts_count: number;
  has_duplicates: boolean;
  warning: string | null;
  distinct_concepts: DistinctPoolConcept[];
  duplicate_concepts?: DistinctPoolConcept[];
  source_entries: PoolSourceEntry[];
  created_at?: string;
}

export interface ContributorAssignmentItem {
  lesson_id: number;
  lesson_title: string;
  lesson_description?: string;
  subject: string;
  session_date?: string;
  group_name?: string;
  session_readiness_status?: string;
  assigned_at: string;
  document_uploaded: boolean;
  document_id: number | null;
  entry_mode?: 'file' | 'text' | string | null;
  report_title?: string | null;
  original_filename: string | null;
  file_size: number | null;
  version?: number | null;
  processing_status: 'awaiting_upload' | 'uploaded' | 'processing' | 'completed' | 'failed' | string;
  error_message: string | null;
  uploaded_at: string | null;
  processed_at: string | null;
}

export interface ContributorUploadHistoryItem {
  id: number;
  document_uuid: string;
  lesson_id: number;
  lesson_title: string;
  subject: string;
  entry_mode?: 'file' | 'text' | string;
  report_title?: string | null;
  original_filename: string;
  file_size: number;
  version: number;
  is_active: boolean;
  processing_status: 'awaiting_upload' | 'uploaded' | 'processing' | 'completed' | 'failed' | string;
  error_message: string | null;
  uploaded_at: string;
  processed_at: string | null;
  report_text?: string;
}

export interface ConceptResult {
  concept: string;
  status: 'recalled' | 'partial' | 'missing' | 'misconception' | string;
  evidence: string;
  explanation: string;
}

export interface C4MatchedConceptItem {
  concept: string;
  normalized_concept: string;
  source_document_ids?: number[];
  evidence?: string;
  explanation?: string;
}

export interface AssessmentDetails {
  assessment_mode?: 'legacy' | 'c4' | string;
  pool_version?: number;
  total_distinct_concepts?: number;
  matched_count?: number;
  c4_rating?: number;
  recall_percentage?: number;
  matched_concepts?: C4MatchedConceptItem[];
  unmatched_concepts?: C4MatchedConceptItem[];
  concept_results?: ConceptResult[];
  missing_concepts?: string[];
  misconceptions?: string[];
  uncertainties?: string[];
  strengths?: string[];
  areas_for_improvement?: string[];
  feedback?: string;
}

export interface Submission {
  id: number;
  lesson_id: number;
  lesson: string;
  subject: string;
  student_id: number;
  student_name: string;
  student_email: string;
  text: string;
  character_count: number;
  submitted_at: string;
  status: string;
  assessment_status: 'received' | 'pending' | 'processing' | 'completed' | 'failed' | string;
  assessment_mode?: 'legacy' | 'c4' | string;
  pool_version?: number | null;
  matched_concepts_count?: number | null;
  total_pool_concepts?: number | null;
  recall_percentage: number | null;
  rating: number | null;
  similarity: number | null;
  feedback: string | null;
  assessment_details?: AssessmentDetails;
}

export interface DashboardAnalyticsResponse {
  role: Role;
  total_sessions: number;
  ready_sessions: number;
  total_submissions: number;
  completed_submissions: number;
  average_rating: number | null;
  rating_distribution: Array<{ rating: string; score: number; count: number }>;
  submission_completion: Array<{ name: string; value: number; status: string }>;
  session_comparison: Array<{
    lesson_id: number;
    session: string;
    full_title: string;
    average_rating: number;
    submissions: number;
    completed: number;
    readiness_status: string;
  }>;
  concept_coverage: Array<{
    lesson_id: number;
    session: string;
    full_title: string;
    avg_matched_concepts: number;
    total_concepts: number;
    coverage_pct: number;
  }>;
  performance_trend: Array<{
    submission_id: number;
    lesson_id: number;
    session: string;
    submitted_date: string;
    rating: number;
    recall_percentage: number;
    matched_concepts: number;
    total_concepts: number;
  }>;
  personal_completion: Array<{ name: string; value: number }>;
  personal_trend: Array<{
    submission_id: number;
    lesson_id: number;
    session: string;
    submitted_date: string;
    rating: number;
    matched_concepts: number;
    total_concepts: number;
  }>;
}

export interface AppNotification {
  id: number;
  title: string;
  message: string;
  category:
    | 'session'
    | 'session_ready'
    | 'contributor_assigned'
    | 'document_uploaded'
    | 'document_processing'
    | 'document_complete'
    | 'document_failed'
    | 'submission'
    | 'assessment_complete'
    | 'assessment_failed'
    | 'info'
    | string;
  link_view?: string;
  link_id?: number;
  is_read: boolean;
  created_at: string;
}

export type Theme = 'neural-dark' | 'aurora-light';
