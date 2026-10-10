output "storage_uri" {
  value = "gs://${google_storage_bucket.outputs.name}"
}

output "image_repo" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "job_service_account" {
  value = google_service_account.job.email
}