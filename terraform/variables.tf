variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region for cluster and artifact registry"
  type        = string
  default     = "us-central1"
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "mlops-gke-cluster"
}

variable "gar_repo_name" {
  description = "Google Artifact Registry repository name"
  type        = string
  default     = "mlops-repo"
}

variable "machine_type" {
  description = "Machine type for GKE worker nodes"
  type        = string
  default     = "e2-standard-2"
}
