// Define the resources

provider "google" {
  project = var.project_id
  region  = var.region

  # The budget API needs a project to charge API calls to when you log in as a user
  user_project_override = true
  billing_project       = var.project_id
}

# Turn on the GCP services this project uses
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "cloudbilling.googleapis.com",
    "billingbudgets.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

# Bucket for all outputs: model, tiles, mosaic (STORAGE_URI points here)
resource "google_storage_bucket" "outputs" {
  name                        = "${var.project_id}-outputs"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  depends_on                  = [google_project_service.apis]
}

# Private Docker registry for the ricemap image
resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = "ricemap"
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

# Identity the Cloud Run job runs as
resource "google_service_account" "job" {
  account_id   = "ricemap-job"
  display_name = "ricemap Cloud Run job"
  depends_on   = [google_project_service.apis]
}

# The job can read and write objects in the outputs bucket, and nothing else
resource "google_storage_bucket_iam_member" "job_outputs" {
  bucket = google_storage_bucket.outputs.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.job.email}"
}

# Project number for the budget filter (read after the APIs are on)
data "google_project" "this" {
  depends_on = [google_project_service.apis]
}

# Email alert at 50%, 90% and 100% of the monthly budget
resource "google_billing_budget" "budget" {
  billing_account = var.billing_account
  display_name    = "ricemap monthly budget"

  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
    # Count usage before free-trial credits, or the alerts never fire during the trial
    credit_types_treatment = "EXCLUDE_ALL_CREDITS"
  }

  amount {
    specified_amount {
      units = tostring(var.budget_amount)
    }
  }

  threshold_rules { threshold_percent = 0.5 }
  threshold_rules { threshold_percent = 0.9 }
  threshold_rules { threshold_percent = 1.0 }

  depends_on = [google_project_service.apis]
}

# Cloud Run Job: 
#runs `ricemap predict-tile`. With no --tile-id, each task picks its tile from CLOUD_RUN_TASK_INDEX.
resource "google_cloud_run_v2_job" "predict" {
  name                = "ricemap-predict"
  location            = var.region
  deletion_protection = false

  template {
    task_count  = 840 # set to 1 for a test run, 840 for the full run
    parallelism = 10 # set to 14 for the full run (GCP trial memory quota 42 GB / 3 GiB)

    template {
      service_account = google_service_account.job.email
      max_retries     = 1
      timeout         = "1800s"

      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/ricemap:${var.image_tag}"
        args  = ["predict-tile"]

        env {
          name  = "STORAGE_URI"
          value = "gs://${google_storage_bucket.outputs.name}"
        }

        resources {
          limits = {
            cpu    = "1"
            memory = "3Gi"
          }
        }
      }
    }
  }

  depends_on = [google_project_service.apis]
}
