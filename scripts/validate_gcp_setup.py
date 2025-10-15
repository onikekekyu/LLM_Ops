import os
from google.cloud import aiplatform
from google.cloud import storage
from google.api_core import exceptions
from dotenv import load_dotenv


def validate_vertex_ai_connectivity() -> bool:
    """
    Validates connectivity to Vertex AI by attempting to initialize a client
    with the project ID and region from environment variables.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        # Get project ID and region from environment variables
        project_id = os.getenv('GCP_PROJECT_ID')
        region = os.getenv('GCP_REGION')
        
        if not project_id or not region:
            print("Error: Missing required environment variables (GCP_PROJECT_ID or GCP_REGION)")
            return False
        
        # Initialize Vertex AI client
        aiplatform.init(
            project=project_id,
            location=region
        )
        
        # If initialization is successful, return True
        print(f"Successfully connected to Vertex AI in project {project_id}, region {region}")
        return True
        
    except Exception as e:
        print(f"Error connecting to Vertex AI: {str(e)}")
        return False

def validate_bucket_access(bucket_name: str) -> bool:
    """
    Validates access to a GCS bucket by attempting to list its contents.
    
    Args:
        bucket_name (str): Name of the GCS bucket to validate
        
    Returns:
        bool: True if bucket is accessible, False otherwise
    """
    try:
        # Initialize storage client
        storage_client = storage.Client()
        
        # Get bucket
        bucket = storage_client.bucket(bucket_name)
        
        # Try to list objects (limited to 1 to minimize API calls)
        next(bucket.list_blobs(max_results=1), None)
        
        print(f"Successfully accessed GCS bucket: {bucket_name}")
        return True
        
    except exceptions.PermissionDenied:
        print(f"Permission denied: Unable to access bucket {bucket_name}")
        return False
    except exceptions.NotFound:
        print(f"Bucket not found: {bucket_name}")
        return False
    except Exception as e:
        print(f"Error accessing bucket: {str(e)}")
        return False

def main():
    """
    Main function to run the validation checks
    """
    # Load environment variables
    load_dotenv()
    
    bucket_name = os.getenv('GCP_BUCKET_NAME')
    if not bucket_name:
        print("Error: GCP_BUCKET_NAME environment variable not set")
        return
    
    # Run validations
    vertex_ai_status = validate_vertex_ai_connectivity()
    bucket_status = validate_bucket_access(bucket_name)
    
    # Print summary
    print("\nValidation Summary:")
    print(f"Vertex AI Connectivity: {'✓' if vertex_ai_status else '✗'}")
    print(f"GCS Bucket Access: {'✓' if bucket_status else '✗'}")

if __name__ == "__main__":
    main()
