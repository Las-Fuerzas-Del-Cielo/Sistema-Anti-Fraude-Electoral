![GitHub Repo Size](https://img.shields.io/github/repo-size/Las-Fuerzas-Del-Cielo/Sistema-Anti-Fraude-Electoral)

# :lion: Anti-Fraud-Electoral-System :lion:

This is an **Open Source** project whose objective is to minimize and detect the possibility of electoral fraud in the upcoming presidential election in Argentina, where the next president will be elected.

The reason for creating this system is to maintain and preserve democracy and transparency for the people of Argentina.

## Index
- [:lion: Anti-Fraud-Electoral-System :lion:](#lion-anti-fraud-electoral-system-lion)
  - [Index](#index)
  - [Objectives](#objectives)
  - [Components](#components)
  - [Repositories and organization](#repositories-and-organization)
  - [Types of Fraud](#types-of-fraud)
    - [Fraudulent Summation](#fraudulent-summation)
    - [Non-Existent Tables](#non-existent-tables)
    - [Correct Table Data Then Incorrect](#correct-table-data-then-incorrect)
    - [False Auditors](#false-auditors)
    - [Judas Auditors](#judas-auditors)
    - [Incompetent Auditors](#incompetent-auditors)
    - [Corrupt Vote Count due to Lack of Auditors](#corrupt-vote-count-due-to-lack-of-auditors)
  - [Users](#users)
  - [Funcionality](#funcionality)
  - [System Arquitecture](#system-architecture)
    - [Main Components](#main-components)
      - [Database](#database)
      - [Backend Services](#backend-services)
      - [Frontend](#frontend)
      - [Batch Processes](#batch-processes)
  - [Responsible](#responsible)
  - [Links of interest](#links-of-interest)
- [How to Contribute](#how-to-contribute)
- [Authors](#authors)
- [Contributors](#contributors)
- [Discord for Developers](#discord-for-developers)

## Objectives

The objectives of this system are:

1. Identify potential electoral fraud.
2. Minimize its occurrence and impact.
3. Speed up its detection to avoid the false declaration of a winner with high levels of fraud/suspicious cases. 

## Components
- Frontend Auditors (data upload)
- Frontend Public (for anyone who wants to access the data)
- Backend (API)

## Repositories and organization
This repository has the purpose of:
- Explain the project in general
- Hosting the [code for the frontend for the auditors](https://github.com/Las-Fuerzas-Del-Cielo/Sistema-Anti-Fraude-Electoral/tree/main/Frontend/fiscales-app-ts).

These repositories are the complementary parts of the project:
- Frontend Public
- [Backend API](https://github.com/Las-Fuerzas-Del-Cielo/api)

## Types of Fraud

What types of fraud are expected to be detected? In an election like this, there are many ways to commit fraud if someone controls the official computer system.

This is the list of types of fraud that we want to attack. That is the backbone of this project, from there all the functionality that is required is going to be built. If you want, you can program some of what is needed for any of the identified types of fraud, if you can think of any other type of fraud and how to minimize it, you can add it to the list and specify what would have to be done for that. And if you have time and energy, you could start by scheduling what is needed in which others arrive to help you.

### Fraudulent Summation

In this case, the Official Computer System has correctly loaded all the data, but even so, the grouping by Province/Nation is incorrect.

### Non-existent Tables

In this case, the Official Computer System has all the data loaded correctly, but in addition to the real tables, there are tables that only exist in the system and that are used to change the global results.

### Correct Table Data Then Incorrect

The third type of fraud would be that the tables loaded into the Official System have different data than the records of the auditors. This can currently be detected by hand if the auditor checks the Official System. The problem is that the official system could show you the correct data for a while and then change it later when the auditor has already consulted it and seen the correct data. Surely an auditor would only check the data once and if it is correct, he would assume that it does not change later.

### False Auditors

These would be people who signed up as auditors only to stay with a table or group of tables and then not go or leave at the last minute when it is very difficult to get replacement and reassign their tables.

The system must have the data of the Table Auditors and their work in the PASO and general elections pre-loaded so that if they were absent without cause, or suspiciously, they will not be called back for the ballotage.

### Judas Auditors

This type of fraud is based on recruiting and enrolling Table Auditors who instead of supervising and defending the votes of LLA, actually do the opposite. The tables with Judas Prosecutors could allow false data to be loaded into the Official System because the accounting was already corrupt.

The system should allow users of the **External Auditors** type to see the data of the Table Auditors in order to investigate them and thus be able to infer whether they are of the Judas type or not. Table Auditors will be invited to optionally provide information of their identities on social networks, and those who do, will score points on their reputation as an Auditors. On the other hand, an army of external auditors can investigate with their ID and those social identities what the real background of those people are and based on the results of these investigations, assign each auditor a score that can be used in an extreme case to disqualify him or at least to observe him with caution.

### Incompetent Auditors  

The system should cover the case of auditors who are simply incompetent and through ignorance upload incorrect data into the system. There should be mechanisms to exclude data from this type of auditor or that some type of user can overwrite them based, for example, on the photographs of the records.

### Corrupt Vote Count due to Lack of Auditors

The system should help to manage the Table Auditors and the General Auditors, so that there is at least one General Auditor in each school as a minimum. Cases where there is not even one General Auditor per school are a giant loss of votes. The allocation of Table Auditors should also be managed, so that in the face of shortages, Auditors are sent to the places where they are most needed. For example, to schools where there are a larger number of electors combined with a smaller number of Table Auditors.

## Users

This app would have these types of users:

1. **Table Auditor:** The main user of this APP would be the Table Auditors of LLA. They would be the ones who upload the data.

2. **General Auditors:** Supervises the Desk Prosecutors at a certain school. The General Auditors MUST take photos of all the acts of all the tables of the school to which they were assigned. In theory there is always a General Auditor, even if there are no Table Auditors. If there are no Table Auditors and the General Auditor has to photograph the acts and upload them to the system, we can have an army of online volunteers that transcribes the values of the photos into numerical records and thus have the first version of data in the system very early after the close of voting. It should be the responsibility of the General Auditor to take the photos of all the tables in a school, because that way we could avoid **Non-Existent Tables** fraud, where in the official system there appear tables that in reality did not exist. If each of our auditors takes a picture of ALL the tables physically present in each school, we could detect those ghost tables.

3. **Party Delegate:** They are trusted people from LLA who during the voting can go from one school to another to support and audit the General Auditors and the Table Auditors.

4. **Internal Auditor:** They would be people from LLA who would analyze the data uploaded by the Table Auditors and compare them with the official data. The APP would try to automate this comparison in order to detect differences and potential fraud.

5. **External Auditor:** This would be anyone who completes the sign up process as an external auditor. This profile would have access to the necessary functionality to consult digitized data and documents and report anomalies that would then be raised by Internal Auditors. It is important to note that fraud has to be detected as soon as possible because once a winner is declared, it will be difficult to overturn no matter how many fraud reports there are afterwards. That eliminates the possibility of just uploading a photo and then having someone at some point type in the data of that photo. In general, the detection of the different types of fraud should be instantaneous and moreover, if possible, anyone from the general population should be able to see the vote count according to the LLA prosecutors even before the official data comes out, so that before a winner is declared there is already a reaction from the general public to a potential large-scale fraud.

6. **Public:** Anyone who wants to see the results online, according to the data uploaded by the Prosecutors of Mesa de LLA. They will also be able to browse all the information available in the system.


## Funcionality

1. **Data Upload**: The APP would allow Table Auditors to upload the data of the tables they supervise. The system would accumulate all the data in a database.

2. **Reports for Auditors**: The system would generate different types of reports aimed at detecting fraud, based on the data uploaded by the Table Prosecutors and Official data.

3. **Consultations for the Public**: The system would allow to execute different types of consultations for the general public.

4. **Map / Report of Auditors Working**: The system should allow to know online where there are and where there aren't auditors, so through networks people can be mobilized to go to audit, especially in the most extreme cases where, for example, there is no one. An online report ordered by severity of where auditors are urgently needed would be optimal. It would be more serious in schools with the largest number of voters where there are the fewest auditors. From there they could take the data that would be updated at all times during the voting of where it is most critical to call through the networks for those schools to be reinforced.

5. **Map / Information of Bunkers-Branches**: The system should allow to visualize a map or several maps in which it should be possible to see where to go to look for personal ballots.

## System Architecture

![image](https://github.com/Las-Fuerzas-Del-Cielo/Sistema-Anti-Fraude-Electoral/assets/140413871/d3b6c243-93b5-41f6-9060-5ab97d90995c)

- **Cloudflare:** Used for DNS, CDN and security management at layer 7.

- **React SPA:** Hosted in S3.

- **Express API:** Hosted as a monolith in a Lambda function. This allows flexibility for developers and avoids the complexity of having to adapt to the development of microservices.

- **Gateway API:** The magic of making a serverless monolith happens here, since all the endpoints are proxied to the Lambda that runs the Express server.

- **DynamoDB:** Another serverless service that avoids us having to deal with scaling configurations and possible unexpected traffic spikes. This ensures that the application can withstand high load levels without crashing.

- **S3 Bucket:** Assets will be uploaded here in the form of videos or images as proof of fraud.

There are many ways to approach the architecture of a system like this. Let's list the most important criteria first, followed by the architecture that will be required.

1. The entire system must be developed and tested in record time. This is the main contraint.

2. We need to put many people to work in parallel, with the minimum friction between them. To achieve this, we must divide the system into blocks of use-cases that interact with each other through well-defined interfaces.

3. We must minimize the trust in each individual who participates, since no one knows each other and no one knows who is who, and some could assume responsibilities with the explicit intention of not fulfilling them, among other things.

4. We must minimize the risk of failures on Election Day, so we must have redundancy not only at the hardware level, but also at the software level.

5. We believe in OPEN SOURCE, PERMISSIONLESS and DECENTRALIZED systems (as far as possible and reasonable for this case). We want to develop a system that not only allows anyone to audit the code because it is open source, but also allows anyone in the world to register with any of the roles/types of users. In this way, for the first time in history, anyone, wherever they are through the Internet, can help audit the election and prevent fraud.

### Main Components 

#### Database

**Main Database**

The system database is, in our case, the SINGLE POINT OF FAILURE (unless it is replicated). We envision having at least one database for the information collected by the Table Auditors and the General Auditors, which will be read/write and from it the information will be used for the functionalities of those roles (Table Auditors, General Auditors).

**Read-Only Database**

For queries from the general public or the online army of auditors, because it is difficult to estimate the number of users in those roles in an open and permissionless system, we may have a replica of the previous database but read-only, or an in-memory or cache version to serve all kinds of query requirements by these types of users.

**User Database**

It would be separated from the rest so that it is built, maintained and operated by people who specialize in System Security and that no one outside that team can break anything here.

#### Backend Services

**Main Backend**

The main backend will be the one with the business logic of the main use-cases, which are those corresponding to the Table Auditors, General Auditors and Party Delegates.

**Backend Read Only**

We may have a backend for the read-only operations of the general public/external auditors to LLA. It is possible that this backend works with an offline replica of the Main Database, updated from time to time.

**Backend for Logins / Signups / User Maintenance**

Normally this would be part of the Main Backend, but since we have so little time, we could separate this group of functionalities for a specialized team to develop this without touching anything on the rest of the system.

#### Frontend

**Web / Mobile UI for Auditors**

The UI for Auditors should be considered mission critical. If it didn't work, we wouldn't have anything, because the auditors are the ones who upload the data that are the basis of all the audits that the system is going to allow to be carried out. Based on the criteria outlined above, to minimize the risk of any module not being ready or not working well, the proposal is to open the field and have multiple developers develop multiple UIs. Then we would publish the links to which the tests we do pass and the rest would be abandoned. Everyone is free to choose the framework and technologies to use for their UI for Auditors, because everyone invests their own time building it. All these UIs would be connected to the Main Backend via a pre-defined API and the defined use-cases could be performed from any of them.

As an extension of the above criterion, it would be optimal if each developer hosted the UI on his own server on his own domain if he wanted. This would make the system more resilient if there were more than one option. This applies to the next UI as well.

If all the polling stations had auditors we are talking about a number of potential users of between 100K and 150K because there are more or less that number of polling stations nationwide.

**Web UI for the general public / external auditors**

The UI for the general public/external auditors and the ideas of non-mission critical functionalities should be a web app. In this case, the potential mass of users is tremendously greater than the previous one, in the order of the 30 or 40 million people who could potentially want to consult the results as they see them and some smaller number than that of people who want to play the role of external auditor and control what the system allows them to control. Allowing any number of people to enter the system to be audited can be the key so that, combined with the use /denunciations through social networks of a large number of people, possible frauds that the people who control the official system (which is a closed and opaque system) could want to do can be discouraged.

In this case, we would also allow any developer to create their own version of this site for the general public and External Auditors, in the technology they want, and then we would publish the links to the sites that pass the tests we make correctly. At the same time, if there were several versions of the site, we would decrease the individual load on each one and lower the risk of not having something working by Election Day.

**Login UI / Signup / User Maintenance**

This would be the specific UI for these use-cases, by people specialized in System Security.

#### Batch Processes 

**Extraction of Data from the Official System**

The official system provides here (https://resultados.mininterior.gob.ar/desarrollo) instructions on how to access certain data through an API. We should have a process that extracts data every so often (5 minutes?) and update our database.

In order to be able to open the database and so that several developers can do different processes using the data from the Official Site plus the data uploaded by the Table Auditors, it is better if there is a process that only extracts the data from the official site and records them in our database. After that process runs every so often, there may be *n* processes, from *n* different developers each looking to detect some different type of fraud.

**Fraud Detection Processes**

With the data uploaded by the Auditors through the mobile app plus the data extracted from the official system, the system has the ability to run multiple processes each specialized in detecting some type of fraud.

The processes that are needed to detect the previously specified types of fraud should be analyzed.

## How to contribute

To make your contribution, you have to create a fork that includes the dev branch and work on it. When you are done with your changes, create a PR from your fork pointing to the dev branch of this repository. If possible, add a detailed description to the PR so that reviewers can get oriented quickly and add the corresponding tags to the changes made.

In summary:
- Create a fork of this repository that includes the **dev** branch.
- Make the changes to the local clone of the fork in the **dev** branch.
- Upload the changes to your fork.
- Create a PR to the **dev** branch of this repository.
- Add a clear description of the changes in the PR.
- Add tags corresponding to the changes in the PR.

## Responsible

In the best Open Source style, anyone who wants to take responsibility for some part of the system can self-list here below, modifying this readme through a PR.

- General Analysis [@Luis-Fernando-Molina](https://www.github.com/Luis-Fernando-Molina)
- UX/UI [@JoseOrtega02](https://github.com/JoseOrtega02), [anyone who wishes to join]
- FrontEnd [@JoseOrtega02](https://github.com/JoseOrtega02), [anyone who wishes to join]

## Links of Interest
- Figma: [Enlace a Figma](https://www.figma.com/file/nyWx6CewFyvb3a7y3g1r7W/Libertarios-APP?type=design&node-id=0%3A1&mode=design&t=L4k93Fh2vw4b8yku-1)
- Trello: [Enlace a Trello](https://trello.com/invite/b/3sdCNjhp/ATTI0ee6e921ed507577043c8411266d7206D016745E/libertarios-app-ux-ui-fronted)

# Authors

- [@Luis-Fernando-Molina](https://www.github.com/Luis-Fernando-Molina)
- [@switTV](https://www.github.com/switTV)

# Contributors
<a href="https://github.com/Las-Fuerzas-Del-Cielo/Sistema-Anti-Fraude-Electoral/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Las-Fuerzas-Del-Cielo/Sistema-Anti-Fraude-Electoral" height="50"/>
</a>

# Discord for Developers

[Discord Link](https://discord.gg/BWDqcpXn)
