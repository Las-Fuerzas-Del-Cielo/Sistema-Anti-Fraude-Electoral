![GitHub Repo Size](https://img.shields.io/github/repo-size/Las-Fuerzas-Del-Cielo/Sistema-Anti-Fraude-Electoral)

# :lion: Electoral Anti-Fraud System :lion:

This is an **Open Source** project with the aim of minimizing and detecting the possibility of electoral fraud in the upcoming presidential elections in Argentina, where the final outcome of the presidency is determined.

The intention behind creating this system is to uphold and preserve democracy and transparency for the Argentine people.

## Table of Contents
- [:lion: Electoral Anti-Fraud System :lion:](#lion-electoral-anti-fraud-system-lion)
  - [Table of Contents](#table-of-contents)
  - [Objectives](#objectives)
  - [Components](#components)
  - [Repositories and Organization](#repositories-and-organization)
  - [Types of Frauds](#types-of-frauds)
    - [Fraudulent Summarization](#fraudulent-summarization)
    - [Non-Existent Polling Stations](#non-existent-polling-stations)
    - [Correct Table Data, Then Incorrect](#correct-table-data-then-incorrect)
    - [False Polling Officials](#false-polling-officials)
    - [Judas Polling Officials](#judas-polling-officials)
    - [Incompetent Polling Officials](#incompetent-polling-officials)
    - [Corrupt Vote Count Due to Missing Polling Official](#corrupt-vote-count-due-to-missing-polling-official)
  - [Users](#users)
  - [Functionality](#functionality)
  - [System Architecture](#system-architecture)
    - [Key Components](#key-components)
      - [Database](#database)
      - [Backend Services](#backend-services)
      - [Frontend](#frontend)
      - [Batch Processes](#batch-processes)
  - [Responsibles](#responsibles)
  - [Useful Links](#useful-links)
- [How to Contribute](#how-to-contribute)
- [Authors](#authors)
- [Contributors](#contributors)

## Objectives

The objectives of this system are:

1. Identify potential electoral frauds.
2. Minimize their occurrence and impact.
3. Expedite their detection to prevent the false declaration of a winner with high levels of fraud/suspicious cases.

## Components
- Frontend for Polling Officials (data entry)
- Public Frontend (for anyone accessing the data)
- Backend (API)

## Repositories and Organization
This repository serves the purpose of:
- Explaining the project in general
- Hosting the [frontend code for polling officials](https://github.com/Las-Fuerzas-Del-Cielo/Sistema-Anti-Fraude-Electoral/tree/main/Frontend/fiscales-app-ts).

These repositories are complementary parts of the project:
- Public Frontend (_TO DO: Include a link when it's created_)
- [Backend API](https://github.com/Las-Fuerzas-Del-Cielo/api)

## Types of Frauds

What types of fraud are expected to be detected? In an election like this, there are many ways to commit fraud if someone controls the official computing system.

This is the list of Types of Frauds that we aim to tackle. This forms the backbone of the project, and all the functionality to be built derives from this. If you like, you can code something for any of the identified types of fraud, and if you think of another type of fraud and how to minimize it, you can add it to the list and specify what needs to be done. If you have the time and energy, you could start coding what's needed while waiting for others to assist.

### Fraudulent Summarization

In this case, the official computing system has all the data correctly loaded, but the grouping by Province/Nation is incorrect.

### Non-Existent Polling Stations

In this case, the official computing system has all the data correctly loaded, but in addition to the real polling stations, there are polling stations that only exist in the system and are used to alter the global results.

### Correct Table Data, Then Incorrect

The third type of fraud would be that the tables loaded into the Official System have different data from the records of the polling officials. Currently, this can be manually detected if the polling official checks the Official System. The problem is that the official system could initially show the correct data for a while and then change it later when the official has already checked and seen it as correct. Most likely, an official would only verify the data once, and if it's correct, they would assume it won't change later.

### False Polling Officials

These would be people who registered as polling officials just to secure a polling station or a group of polling stations and then not show up or abandon at the last moment when it's very difficult to find a replacement and reassign their stations.

The system should have pre-loaded data on the Polling Officials and their work in the PASO and GENERAL elections so that if they were absent without cause or in a suspicious manner, they wouldn't be called for the BALLOTAGE.

### Judas Polling Officials

This type of fraud is based on recruiting and enrolling Polling Officials who, instead of overseeing and defending the votes of LLA, actually do the opposite. Polling stations with Judas Polling Officials could allow false data to be entered into the Official System because the counting has already been corrupted.

The system should allow users of the **External Auditors** type to view the data of the Polling Officials in order to investigate them and determine if they are of the Judas type or not. Polling Officials will be invited to provide optional information about their identities on social networks, and those who do so will earn points in their reputation as Polling Officials. On the other hand, an army of external auditors can investigate with their ID and these social identities what the real background of these people is, and based on the results of these investigations, assign a score to each polling official that can be used in an extreme case to disqualify them or at least observe them with caution.

### Incompetent Polling Officials

The system should cover cases of polling officials who are simply incompetent and, due to ignorance, enter data incorrectly into the system. This means that there should be mechanisms to exclude data from these polling officials or allow some type of user to override them based on, for example, photographs of the records.

### Corrupt Vote Count Due to Missing Polling Official

The system should help manage the Polling Officials and General Polling Officials so that there is at least one General Polling Official in each school at a minimum. Cases where there is not even one General Polling Official per school result in a huge loss of votes. It should also manage the assignment of Polling Officials so that in case of shortages, they are sent to the places where they are most needed. For example, to schools with a higher number of voters combined with a lower number of Polling Officials.

## Users

This APP would have these types of users:

1. **Polling Official:** The primary user of this APP would be the Polling Officials of LLA. They would be responsible for entering the data.

2. **General Polling Official:** Oversees the Polling Officials in a specific school. The General Polling Official MUST take photos of all the records from all the polling stations in the assigned school. In theory, there should always be at least one General Polling Official, even if there might not be any Polling Officials. If they do this and upload the data to the system, we can have an online army of volunteers who transcribe the values from the photos into numerical records, providing the first version of data into the system shortly after the voting ends. It should be the responsibility of the General Polling Official to take photos of all the physically present polling stations in each school because that way we can detect those phantom polling stations in the Official System.

3. **Party Delegate:** These are trusted individuals of LLA who can go from one school to another during the voting to support and audit the General Polling Officials and Polling Officials.

4. **Internal Auditor:** These would be people from LLA who would analyze the data entered by the Polling Officials and compare it with the official data. The APP should aim to automate this comparison to detect differences and potential frauds.

5. **External Auditor:** Any individual who completes the sign-up process as an external auditor. This profile would have access to the necessary functionality to query data and digitized documents and report anomalies that would then be investigated by Internal Auditors. It's important to note that fraud needs to be detected as early as possible because once someone is declared a winner, it's unlikely to be reversed, regardless of the number of fraud reports that come afterward. This eliminates the possibility of just uploading a photo, and someone at some point entering the data from that photo. In general, the detection of different types of fraud should be instantaneous, and, if possible, anyone from the general population should be able to view the vote count according to the LLA Polling Officials even before the official data is released, so that there's a public reaction to a potential large-scale fraud before a winner is declared.

6. **Public:** Anyone who wants to view the results online, based on the data entered by the LLA Polling Officials. They can also navigate through all the information available in the system.

## Functionality

1. **Data Entry**: The APP would allow Polling Officials to enter data from the polling stations they supervise. The system would accumulate all the data in a database.

2. **Reports for Auditors**: The system would generate different types of reports aimed at detecting fraud, based on the data entered by the Polling Officials and the Official Data.

3. **Queries for the Public**: The system would allow various types of queries for the general public.

4. **Map / Report of Polling Officials Working**: The system should allow knowing online where there are and where there are no Polling Officials, so through social networks, people can be mobilized to go and oversee the voting, especially in the most extreme cases where, for example, there are none. An online report sorted by severity of where Polling Officials are urgently needed would be optimal. It would be more critical in schools with a higher number of voters and fewer Polling Officials. From there, data could be taken, which would be updated at all times during the voting, on where it's most critical to call for reinforcements through social networks.

5. **Map / Information on Bunkers-Branches**: The system should allow visualizing a map or multiple maps where people can go to get their personal ballots, so people can see where to get them.

## System Architecture

![image](https://github.com/Las-Fuerzas-Del-Cielo/Sistema-Anti-Fraude-Electoral/assets/140413871/d3b6c243-93b5-41f6-9060-5ab97d90995c)

- **Cloudflare:** Used for DNS management, CDN, and security at layer 7.

- **React SPA:** Hosted on S3.

- **Express API:** Hosted as a monolith in a Lambda function. This provides flexibility for developers and avoids the complexity of adapting to microservices development.

- **API Gateway:** This is where the magic of creating a serverless monolith happens, as all endpoints are proxied to the Lambda that runs the Express server.

- **DynamoDB:** Another serverless service that eliminates the need to deal with scaling configurations and potential unexpected traffic spikes. This ensures that the application can handle high levels of load without failing.

- **S3 Bucket:** This is where assets in the form of videos or images will be uploaded as evidence of fraud.

There are many ways to approach the architecture of a system like this. Let's first list the most important criteria we want to follow, and from there, we will derive the emerging architecture.

1. The entire system must be developed and tested in record time. That is the main constraint.

2. We need to put many people to work in parallel with minimal friction between them. To achieve this, we must divide the system into blocks of use cases that interact with each other through well-defined interfaces.

3. We must minimize trust in each individual who participates because nobody knows each other, and some might assume responsibilities with the explicit intention of not fulfilling them, among other things.

4. We must minimize the risk of failures on election day, so we need redundancy not only at the hardware level but also at the software level.

5. We believe in OPEN SOURCE, PERMISSIONLESS, and DECENTRALIZED systems (as far as possible and reasonable for this case). We want to develop a system that not only allows anyone to audit its code because it is open source but also allows anyone in the world to register with any of its roles/user types. This way, for the first time in history, anyone, wherever they are via the Internet, can help audit the election and prevent fraud.

### Key Components

#### Database

**Main Database**

The system's main database is, in our case, the SINGLE POINT OF FAILURE (unless it's replicated). We envision having at least one database for the information collected by the Table Officers and General Officers, which will be read/write, and from it, the information will be served for the functionalities of those roles (Table Officers, General Officers).

**Read-Only Database**

For queries from the General Public or the online army of auditors, given that it is difficult to estimate the number of users in these roles in an open and permissionless system, we may have a replica of the previous database but in read-only mode, or an in-memory or cache version to serve all types of query requests from these types of users.

**User Database**

It would be separated from the rest so that it is built, maintained, and operated by specialized people in Systems Security, and no one outside that team can break anything here.

#### Backend Services

**Main Backend**

The main backend will have the business logic of the main use cases, corresponding to Table Officers, General Officers, Party Delegates.

**Read-Only Backend**

It is possible that we have a backend for read-only operations by the general public / external auditors of LLA. This backend might work with an offline replica of the Main Database, updated from time to time.

**Backend for Logins / Signups / User Maintenance**

Normally, this would be part of the Main Backend, but since we have so little time, we could separate this group of functionalities so that a specialized team develops it without touching anything else in the system.

#### Frontend

**Web / Mobile UI for Table Officers**

The UI for Table Officers should be considered mission-critical. If it doesn't work, we have nothing because Table Officers are the ones who enter the data, which is the basis for all the audits that the system will allow. Based on the previously mentioned criteria of minimizing the risk of a module not being ready or not functioning well, the proposal is to open the field and allow multiple developers to create multiple UIs. We would then publish links to those that pass the tests we conduct, and the rest would be abandoned. Each person is free to choose the framework and technologies to use for their Table Officer UI because everyone invests their own time in building it. All these UIs would connect to the Main Backend via a predefined API, and from any of them, the defined use cases can be performed.

As an extension of the previous criterion, it would be even better if each developer hosts the same thing on their own server for their Table Officer UI, including their own domain if they wish. This would make the system more resilient if there were more than one option. This applies to the following UI as well.

If all tables have Table Officers, we are talking about a potential number of users between 100K and 150K because there are approximately that many polling stations nationwide.

**Web UI for the general public / external auditors**

The UI for the general public / external auditors and the ideas for non-critical functionality should be a web app. In this case, the potential user base is significantly larger than the previous one, in the order of 30 to 40 million people who may want to check the results as LLA sees them, and a smaller number of people who want to play the role of an external auditor and control what the system allows them to control. Allowing any number of people to enter the system for auditing can be the key to, combined with the use/reports through social networks by a large number of people, discouraging potential frauds that the people who control the official system (which is a closed and opaque system) may want to commit.

In this case, we would also allow any developer to create their own version of this site for the general public and external auditors in the technology they prefer, and then we would publish links to the sites that pass our tests. At the same time, if there were several versions of the site, we would reduce the individual load on each one and lower the risk of not having something working on election day.

**UI for Login / Signup / User Maintenance**

This would be the specific UI for these use cases, handled by people specialized in Systems Security.

#### Batch Processes

**Extraction of Data from the Official System**

The official system provides instructions here (https://resultados.mininterior.gob.ar/desarrollo) on how to access certain data from it through an API. We should have a process that extracts this data at regular intervals (5 minutes?) and updates our database.

To open the field for various developers to create different processes using data from the Official System plus the data uploaded by Table Officers, it's better if there is a process that only extracts the data from the official site and stores it in our database. After running this process at regular intervals, there can be multiple processes, each specialized in detecting a different type of fraud.

The processes needed to detect the previously specified types of fraud should be analyzed.

## How to Contribute

To make your contribution, you need to create a fork that includes the dev branch and work on it. When you've finished your changes, create a PR from your fork pointing to the dev branch of this repository. If possible, add a detailed description to the PR so that reviewers can quickly orient themselves and add the appropriate labels to the changes made.

In summary:
- Create a fork of this repository that includes the **dev** branch.
- Make changes in the local clone of the fork on the **dev** branch.
- Upload the changes to your fork.
- Create a PR towards the **dev** branch of this repository.
- Add a clear description of the changes in the PR.
- Add labels corresponding to the changes in the PR.

## Responsible Parties

In an open-source manner, anyone who wishes to take responsibility for a part of the system can list themselves below by modifying this readme through a PR.

- General Analysis [@Luis-Fernando-Molina](https://www.github.com/Luis-Fernando-Molina)
- UX/UI [@JoseOrtega02](https://github.com/JoseOrtega02), [anyone who wants to join]
- Frontend [@JoseOrtega02](https://github.com/JoseOrtega02), [anyone who wants to join]

## Useful Links
- Figma: [Figma Link](https://www.figma.com/file/nyWx6CewFyvb3a7y3g1r7W/Libertarios-APP?type=design&node-id=0%3A1&mode=design&t=L4k93Fh2vw4b8yku-1)
- Trello: [Trello Link](https://trello.com/invite/b/3sdCNjhp/ATTI0ee6e921ed507577043c8411266d7206D016745E/libertarios-app-ux-ui-fronted)

# Authors

- [@Luis-Fernando-Molina](https://www.github.com/Luis-Fernando-Molina)
- [@switTV](https://www.github.com/switTV)

# Collaborators
<a href="https://github.com/Las-Fuerzas-Del-Cielo/Sistema-Anti-Fraude-Electoral/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Las-Fuerzas-Del-Cielo/Sistema-Anti-Fraude-Electoral" height="50"/>
</a>

# Developer Discord

[Discord Link](https://discord.gg/BWDqcpXn)

[Enlace al Discord](https://discord.gg/BWDqcpXn)