# MLOps team 39 

## Step-by-Step Guide for Using DVC in the Project

## To retrieve all files our team is tracking with DVC, you could:

1. Clone the repository from GitHub:

```bash
git clone https://github.com/marcelarosalesj/mna_mlops_equipo39
cd mna_mlops_equipo39
```

2. Install DVC if needed
```bash
pip install dvc

```

Initialize DVC
```bash
dvc init

```

3. Configure DVC Remote (this only needs to be done once)

Imstall DVC drive if needed 
```bash
pip install 'dvc[gdrive]'
```

```bash
dvc remote add -d gdrive_storage gdrive://1XDzpeNIrwyg9sMHpmKM5Z4eDxrdRlPEG
```

4. Add the Gdrive credentials

```bash
dvc remote modify gdrive_storage gdrive_client_id '647670916535-0t026mcb74agt85fnq1njm95vn796slt.apps.googleusercontent.com' --local
dvc remote modify gdrive_storage gdrive_client_secret '' --local
```

Check DVC Remote Configuration (Google connection)
Ensure that your DVC remote (Google Drive) is properly set up. You can check this with:
```bash
dvc remote list
```


5. You can proceed to pull the DVC-tracked files from the remote storage (Google Drive)
```bash
dvc pull

```


## Add a new file to DVC 
Suppose you have a new dataset students_perf_data_test_dvc.csv. You can track it with DVC:

```bash
dvc add students_perf_data_test_dvc.csv
```
This command generates a .dvc file (e.g., students_perf_data_test_dvc.csv.dvc) that links the dataset with DVC.

Stage and commit to Git: After adding the file to DVC, you need to commit the changes to Git:

```bash
git add students_perf_data_test_dvc.csv.dvc .gitignore
git commit -m "Track new test dataset with DVC"
```
The dataset itself is not stored in Git, only the metadata (.dvc file) that tells DVC where to find the dataset in Google Drive.

Push the metadata to GitHub:
```bash
git remote add origin https://github.com/marcelarosalesj/mna_mlops_equipo39
git push -u origin main
```

Push data to remote (Google Drive): To upload the dataset to Google Drive, use:
```bash
dvc push
```

Modify the tracked file: If you modify the dataset and want to track the changes:

Modify the file students_perf_data_test_dvc.csv.
Run:
```bash
dvc add students_perf_data_test_dvc.csv
git add students_perf_data_test_dvc.csv.dvc
git commit -m "Updated dataset with new records"
git push
dvc push
```
This will create a new version of the dataset, tracked by DVC. The dataset itself is not stored in Git, only the metadata (.dvc file) that tells DVC where to find the dataset in Google Drive.
DVC push stores the dataset in google drive.

## Track versions of an existing file

1. Verify existing versions
For example, to check the history of students_perf_data_test_dvc.csv, use:

```bash 
git log -- students_perf_data_test_dvc.csv.dvc
```
2. Checkout a Previous Version of the Dataset 
After identifying the specific commit hash of the version you want to retrieve, you can check out that version of the .dvc file and use DVC to restore the corresponding data.

Checkout the previous version of the .dvc file:

```bash
git checkout <commit-hash> <dataset>.dvc
```
In our example:

```bash
git checkout <commit-hash> students_perf_data_test_dvc.csv.dvc
```
Run DVC to retrieve the corresponding data from Google Drive:

```bash
dvc checkout
```

This command ensures that the dataset version linked to the specific .dvc file is pulled from the remote storage.


## Appendix: Setting Up DVC with Google Drive
This section outlines how DVC was configured to use Google Drive as the remote storage for datasets.

Step-by-Step DVC Setup with Google Drive
1. Install DVC and the Google Drive Plugin
First, make sure DVC and the Google Drive plugin are installed in your environment:

```bash
pip install dvc
pip install 'dvc[gdrive]'
```

2. Initialize DVC in the Project
Initialize DVC in the local repository (this step only needs to be done once, typically by the person setting up the project):

```bash
dvc init
```
This creates a .dvc/ folder in the repository, which contains metadata for managing datasets.

3. Configure Google Drive as the Remote Storage
Set up Google Drive as the default remote for storing datasets. This allows DVC to push datasets to Google Drive and pull them later. Run the following command to configure the remote:

```bash
dvc remote add -d gdrive_storage gdrive://<Google-Drive-Folder-ID>
```
Replace <Google-Drive-Folder-ID> with the actual ID of the folder you want to use in Google Drive.

4. Set Google Drive Client Credentials
To allow DVC to access Google Drive, you need to provide your Google OAuth Client ID and Client Secret. These are required for authentication and authorizing DVC to push and pull datasets from Google Drive.

Run the following commands to add your Google Drive credentials:

```bash
dvc remote modify gdrive_storage gdrive_client_id 'your-client-id'
dvc remote modify gdrive_storage gdrive_client_secret 'your-client-secret'
```
Replace 'your-client-id' with the actual Client ID.
Replace 'your-client-secret' with the actual Client Secret.

To allow Google Drive storage you need to fololow the instructions outlined in the DVC documentation:
[Documentation here](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended)

The documentation explains how to set an OAuth connection through google drive API.