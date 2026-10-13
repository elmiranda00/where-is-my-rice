// Inputs

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "Region for the bucket, registry and job (preferred close to Planetary Computer, West Europe)"
  type        = string
  default     = "europe-west4"
}

variable "billing_account" {
  description = "Billing account ID, from: gcloud billing accounts list"
  type        = string
}

variable "budget_amount" {
  description = "Monthly budget, in the billing account's currency"
  type        = number
  default     = 50
}

variable "image_tag" {
  description = "Tag of the ricemap image in Artifact Registry"
  type        = string
  default     = "v3"
}