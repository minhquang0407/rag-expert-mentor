# Trade-Offs in Data Systems Architecture

*There are no solutions; there are only trade-offs. [...] But you try to get the best trade-off you can get, and that's all you can hope for.*

—Thomas Sowell, interview with Fred Barnes (2005)

Data is central to much application development today. With web and mobile apps, software as a service (SaaS), and cloud services, it has become normal to store data from many different users in a shared server-based data infrastructure. Data from user activity, business transactions, devices, and sensors needs to be stored and made available for analysis. As users interact with an application, they both read the data that is stored and generate more data.

Small amounts of data, which can be stored and processed on a single machine, are often fairly easy to deal with. However, as the data volume or the rate of queries grows, it needs to be distributed across multiple machines, which introduces many challenges. As the needs of the application become more complex, it is no longer sufficient to store everything in one system, and it might be necessary to combine multiple storage or processing systems that provide different capabilities.

We call an application *data-intensive* if data management is one of the primary challenges in developing the application [1]. While in *compute-intensive* systems the challenge is parallelizing a very large computation, in data-intensive applications we usually worry more about things like storing and processing large data volumes, managing changes to data, ensuring consistency in the face of failures and concurrency, and making sure services are highly available.

1

Such applications are typically built from standard building blocks that provide commonly needed functionality. For example, many applications need to do the following:

- Store data so that they, or another application, can find it again later (*databases*)
- Remember the result of an expensive operation, to speed up reads (*caches*)
- Allow users to search data by keyword or filter it in various ways (*search indexes*)
- Handle events and data changes as soon as they occur (*stream processing*)
- Periodically crunch a large amount of accumulated data (*batch processing*)

In building an application we typically take several software systems or services, such as databases or APIs, and glue them together with application code. If you are doing exactly what the data systems were designed for, this process can be quite easy.

However, as your application becomes more ambitious, challenges arise. There are many database systems with different characteristics, suitable for different purposes—how do you choose which one to use? There are various approaches to caching, several ways of building search indexes, and so on—how do you reason about their trade-offs? You need to figure out which tools and which approaches are the most appropriate for the task at hand, and it can be difficult to combine tools when you need to do something that a single tool cannot do alone.

This book is a guide to help you make decisions about which technologies to use and how to combine them. As you will see, no one approach is fundamentally better than others; everything has pros and cons. With this book, you will learn to ask the right questions to evaluate and compare data systems so that you can figure out which approach will best serve the needs of your particular application.

We will start our journey by looking at some of the ways that data is typically used in organizations today. Many of the ideas here have their origin in *enterprise software* (i.e., the software needs and engineering practices of large organizations, such as big corporations and governments), since historically, only large organizations had the large data volumes that required sophisticated technical solutions. If your data volume is small enough, you can simply keep it in a spreadsheet! However, more recently it has also become common for smaller companies and startups to manage large data volumes and build data-intensive systems.

One of the key challenges with data systems is that different people need to do very different things with data. If you are working at a company, you and your team will have one set of priorities, while another team may have entirely different goals, even though you might be working with the same dataset! Moreover, those goals might not be explicitly articulated, which can lead to misunderstandings and disagreement about the right approach.

2 | Chapter 1: Trade-Offs in Data Systems Architecture

To help you understand your choices, this chapter compares several contrasting concepts and explores their trade-offs. We will consider the following topics:

- The difference between operational and analytical systems (“**Operational Versus Analytical Systems**” on page 3)
- The pros and cons of cloud services and self-hosted systems (“**Cloud Versus Self-Hosting**” on page 12)
- When to move from single-node systems to distributed systems (“**Distributed Versus Single-Node Systems**” on page 19)
- Balancing the needs of the business and the rights of the user (“**Data Systems, Law, and Society**” on page 24)

This chapter also defines terminology that you will need for the rest of the book.

### **Terminology: Frontends and Backends**

Much of what we will discuss in this book relates to *backend development*. To explain that term: for web applications, the client-side code (which runs in a web browser) is called the *frontend*, and the server-side code that handles user requests is known as the *backend*. Mobile apps are similar to frontends in that they provide user interfaces, which often communicate over the internet with a server-side backend. Frontends sometimes manage data locally on the user's device [2], but the greatest data infrastructure challenges commonly lie in the backend: a frontend needs to handle only one user's data, whereas the backend manages data on behalf of *all* the users.

A backend service is often reachable via HTTP (or sometimes WebSocket); it usually consists of application code that reads and writes data in one or more databases and sometimes interfaces with additional data systems, such as caches or message queues (which we might collectively call *data infrastructure*). The application code is often *stateless* (i.e., when it finishes handling one HTTP request, it forgets everything about that request), and any information that needs to persist from one request to another needs to be stored either on the client or in the server-side data infrastructure.

## **Operational Versus Analytical Systems**

If you are working on data systems in an enterprise, you are likely to encounter several different types of people who work with data. The first type are *backend engineers* who build services that handle requests for reading and updating data; these services often serve external users, either directly or indirectly via other services (see “**Microservices and Serverless**” on page 21). Sometimes services are for internal use by other parts of the organization.

Operational Versus Analytical Systems | 3

In addition to the teams managing backend services, two other groups of people typically require access to an organization's data: *business analysts*, who generate reports about the activities of the organization to help management make better decisions (*business intelligence*, or BI), and *data scientists*, who look for novel insights in data or who create user-facing product features that are enabled by data analysis and machine learning (ML)/AI (e.g., "people who bought X also bought Y" recommendations on an ecommerce website, predictive analytics such as risk scoring or spam filtering, and ranking of search results).

Although business analysts and data scientists tend to use different tools and operate in different ways, they have some practices in common. First, both perform *analytics*, which means they look at the data that the users and backend services have generated. Second, they generally do not modify this data (except perhaps for fixing mistakes), although they might create derived datasets in which the original data has been processed in some way.

This has led to a split between two types of systems—a distinction that we will use throughout this book:

- *Operational systems* consist of the backend services and data infrastructure where data is created—for example, by serving external users. Here, the application code both reads and modifies the data in its databases, based on the actions performed by the users.
- *Analytical systems* serve the needs of business analysts and data scientists. They contain a read-only copy of the data from the operational systems, and they are optimized for the types of data processing that are needed for analytics.

As we shall see in the next section, operational and analytical systems are often kept separate, for good reasons. As these systems have matured, two new specialized roles have emerged: data engineers and analytics engineers. *Data engineers* are the people who know how to integrate the operational and analytical systems and who take responsibility for the organization's data infrastructure more widely [3]. *Analytics engineers* model and transform data to make it more useful for the business analysts and data scientists in an organization [4].

Many engineers specialize in either the operational or the analytical side. However, this book covers both operational and analytical data systems, since both play an important role in the lifecycle of data within an organization. We will explore in depth the data infrastructure that is used to deliver services to both internal and external users so that you can work better with your colleagues on the other side of this divide.

4 | Chapter 1: Trade-Offs in Data Systems Architecture

### Characterizing Transaction Processing and Analytics

In the early days of business data processing, a write to the database typically corresponded to a commercial transaction taking place: making a sale, placing an order with a supplier, paying an employee's salary, etc. As databases expanded into areas that didn't involve money changing hands, the term *transaction* nevertheless stuck, referring to a group of reads and writes that form a logical unit.

**Chapter 8** explores in detail what we mean by a transaction. This chapter uses the term loosely to refer to low-latency reads and writes.

Even though databases started being used for many kinds of data—posts on social media, moves in a game, contacts in an address book, and much, much more—the basic access pattern remained similar to processing business transactions. An operational system typically looks up a small number of records by a key (this is called a *point query*). Records are inserted, updated, or deleted based on the user's input. Because these applications are interactive, this access pattern became known as *online transaction processing* (OLTP).

However, databases also started being increasingly used for analytics, which has very different access patterns compared to OLTP. Usually, an analytical query scans over a huge number of records and calculates aggregate statistics (such as count, sum, or average) rather than returning the individual records to the user. For example, a business analyst at a supermarket chain may want to answer analytical queries such as these:

- What was the total revenue of each of our stores in January?
- How many more bananas than usual did we sell during our latest promotion?
- Which brand of baby food is most often purchased together with brand X diapers?

The reports that result from these types of queries are important for BI, helping management decide what to do next. To differentiate this pattern of using databases from transaction processing, it has been called *online analytical processing* (OLAP) [5]. The difference between OLTP and analytics is not always clear-cut, but some typical characteristics are listed in **Table 1-1**.

Operational Versus Analytical Systems | 5

*Table 1-1. Comparing characteristics of operational and analytical systems*

| <b>Property</b>     | <b>Operational systems (OLTP)</b>               | <b>Analytical systems (OLAP)</b>          |
|---------------------|-------------------------------------------------|-------------------------------------------|
| Main read pattern   | Point queries (fetch individual records by key) | Aggregate over large number of records    |
| Main write pattern  | Create, update, and delete individual records   | Bulk import (ETL) or event stream         |
| Human user example  | End user of web/mobile application              | Internal analyst, for decision support    |
| Machine use example | Checking if an action is authorized             | Detecting fraud/abuse patterns            |
| Type of queries     | Fixed, predefined by application                | Arbitrary, ad-hoc exploration by analysts |
| Query volume        | Lots of small queries                           | Few queries, each is complex              |
| Data represents     | Latest state of data (current point in time)    | History of events that happened over time |
| Dataset size        | Gigabytes to terabytes                          | Terabytes to petabytes                    |

The meaning of *online* in OLAP is unclear; it probably indicates that queries are not just for predefined reports, but that analysts use the OLAP system interactively for explorative queries.

With operational systems, users are generally not allowed to construct custom SQL queries and run them on the database, since that would potentially allow them to read or modify data that they do not have permission to access. They might also write queries that are expensive to execute and hence affect the database performance for other users. For these reasons, OLTP systems mostly run fixed sets of queries that are baked into the application code, with one-off custom queries used only occasionally for maintenance or troubleshooting. On the other hand, analytical databases usually give their users the freedom to write arbitrary SQL queries by hand, or to generate queries automatically using a data visualization or dashboard tool such as Tableau, Looker, or Microsoft Power BI.

Another type of system is designed for analytical workloads (queries that aggregate over many records) but embedded into user-facing products. Systems designed for this type of use, known as *product analytics* or *real-time analytics*, include Pinot, Druid, and ClickHouse [6]. Such systems ingest data in real time and are optimized for low-latency query responses. In contrast, traditional OLAP systems typically ingest data in batches and are optimized for high-throughput query processing.

6 | Chapter 1: Trade-Offs in Data Systems Architecture

### Data Warehousing

At first, the same databases were used for both transaction processing and analytical queries. SQL turned out to be quite flexible in this regard; it works well for both types of queries. In the late 1980s and early 1990s, however, a trend arose for companies to stop using their OLTP systems for analytics purposes and to run the analytics on a separate database system instead. This separate database was called a *data warehouse*.

A large enterprise may have dozens, even hundreds, of OLTP systems: systems powering the customer-facing website, controlling point-of-sale (checkout) systems in physical stores, tracking inventory in warehouses, planning routes for vehicles, managing suppliers, administering employees, and performing many other tasks. Each of these systems is complex and needs a team of people to maintain it, so they end up operating mostly independently from one another.

It is usually undesirable for business analysts and data scientists to directly query these OLTP systems, for several reasons:

- The data of interest may be spread across multiple operational systems, making it difficult to combine those datasets in a single query (a problem known as *data silos*).
- The kinds of schemas and data layouts that are good for OLTP are less well suited for analytics (see “**Stars and Snowflakes: Schemas for Analytics**” on page 77).
- Analytical queries can be quite expensive, and running them on an OLTP database would impact the performance for other users.
- The OLTP systems might reside in a separate network that users are not allowed to directly access, for security or compliance reasons.

A *data warehouse*, by contrast, is a separate database that analysts can query to their hearts' content, without affecting OLTP operations [7]. As we shall see in **Chapter 4**, data warehouses often store data very differently from OLTP databases, to optimize for the types of queries that are common in analytics.

The data warehouse contains a read-only copy of the data from all the various OLTP systems in the company. Data is extracted from OLTP databases (using either a periodic data dump or a continuous stream of updates), transformed into an analysis-friendly schema, cleaned up, and then loaded into the data warehouse. This process of getting data into the data warehouse is known as *extract-transform-load* (ETL) and is illustrated in **Figure 1-1**. Sometimes the order of the *transform* and *load* steps is swapped (i.e., the transformation is done in the data warehouse, after loading), resulting in *ELT*.

Operational Versus Analytical Systems | 7

*Figure 1-1. A simplified outline of ETL into a data warehouse*

In some cases, the data sources of the ETL processes are external SaaS products such as customer relationship management (CRM), email marketing, or credit card processing systems. In those cases, you do not have direct access to the original database, since it is accessible only via the software vendor's API. Bringing the data from these external systems into your own data warehouse can enable analyses that are not possible via the SaaS API. ETL for SaaS APIs is often implemented by specialist data connector services such as Fivetran, Singer, or Airbyte.

Some database systems offer *hybrid transactional/analytical processing* (HTAP), which aims to enable OLTP and analytics in a single system without requiring ETL from one system into another [8, 9]. However, many HTAP systems internally consist of an OLTP system coupled with a separate analytical system, hidden behind a common interface—so the distinction between the two remains important for understanding how these systems work.

Moreover, even though HTAP exists, it is common to have a separation between transactional and analytical systems because of their different goals and requirements. In particular, it is considered good practice for each operational system to have its own database (see *“Microservices and Serverless” on page 21*), leading to potentially hundreds of separate operational databases; on the other hand, an

8 | Chapter 1: Trade-Offs in Data Systems Architecture

enterprise usually has a single data warehouse, so that business analysts can combine data from several operational systems in a single query.

HTAP, therefore, does not replace data warehouses. Rather, it is useful when the same application needs to both perform analytical queries that scan a large number of rows and read and update individual records with low latency. Fraud detection can involve such workloads, for example [10].

The separation between operational and analytical systems is part of a wider trend. As workloads have become more demanding, systems have become more specialized and optimized for particular workloads. General-purpose systems can handle small data volumes comfortably, but the greater the scale, the more specialized systems tend to become [11].

#### **From data warehouse to data lake**

A data warehouse often uses a *relational* data model that is queried through SQL (see [Chapter 3](#)), perhaps using specialized BI software. This model works well for the types of queries that business analysts need to make, but it is less well suited to the needs of data scientists performing tasks such as these:

- Transforming data into a form that is suitable for training an ML model. This often requires turning the rows and columns of a database table into a vector or matrix of numerical values called *features*. The process of performing this transformation in a way that maximizes the performance of the trained model is called *feature engineering*, and it commonly requires custom code that is difficult to express using SQL.
- Using natural language processing (NLP) techniques on textual data (e.g., reviews of a product) to try to extract structured information from it (e.g., the sentiment of the author, or which topics they mention). Similarly, data scientists might need to extract structured information from photos by using computer vision techniques.

Although there have been efforts to add ML operators to a SQL data model [12] and to build efficient ML systems on top of a relational foundation [13], many data scientists prefer not to work in a relational database such as a data warehouse. Instead, many prefer to use Python data analysis libraries such as Pandas and scikit-learn, statistical analysis languages such as R, and distributed analytics frameworks such as Spark [14]. We discuss these further in [“DataFrames, Matrices, and Arrays” on page 105](#).

Consequently, organizations face a need to make data available in a form that is suitable for use by data scientists. The answer is a *data lake*: a centralized data repository that holds a copy of any data that might be useful for analysis, obtained from operational systems via ETL processes. The difference from a data warehouse is that a data lake simply contains files, without imposing any particular file format, data

Operational Versus Analytical Systems | 9

model, or schema [15]. Files in a data lake might be collections of database records, encoded using a file format such as Avro or Parquet (see **Chapter 5**), but a data lake can equally well contain text, images, videos, sensor readings, sparse matrices, feature vectors, genome sequences, or any other kind of data [16]. Besides being more flexible, a data lake is also often cheaper than relational data storage, since it can use commoditized file storage such as object stores (see “**Cloud Native System Architecture**” on page 14).

ETL processes have been generalized to *data pipelines*, and in some cases the data lake has become an intermediate stop on the path from the operational systems to the data warehouse. The data lake contains data in the “raw” form produced by the operational systems, without the transformation into a relational data warehouse schema. This approach has the advantage that each consumer of the data can transform the raw data into the form that best suits their needs. It’s sometimes called the *sushi principle*: “raw data is better” [17].

#### **Beyond the data lake**

As analytics practices have matured, organizations have been increasingly paying attention to the management and operations of analytical systems and data pipelines, as captured, for example, in the DataOps Manifesto [18]. This has been driven partly by issues of governance, privacy, and compliance with regulations such as the General Data Protection Regulation (GDPR) and California Consumer Privacy Act (CCPA), which we discuss in “**Data Systems, Law, and Society**” on page 24 and in **Chapter 14**.

Another important factor is that data for analytics is increasingly made available not only as files and relational tables, but as streams of events (see **Chapter 12**). With file-based data analysis, you can rerun the analysis periodically (e.g., daily) to respond to changes in the data, but stream processing allows analytical systems to respond to events much faster, on the order of seconds. Depending on the application and its time-sensitivity, a stream processing approach can be valuable, for example, to identify and block potentially fraudulent or abusive activity.

In some cases the outputs of analytical systems are made available to operational systems (a process sometimes known as *reverse ETL* [19]). For example, an ML model that was trained on data in an analytical system may be deployed to production so that it can generate recommendations for end users, such as “people who bought X also bought Y.” Machine learning models can be deployed to operational systems by using specialized tools such as TFX, Kubeflow, or MLflow.

### **Systems of Record and Derived Data**

Related to the distinction between operational and analytical systems, this book also distinguishes between *systems of record* and *derived data systems*. These terms are useful because they help clarify the flow of data through a system:

10 | Chapter 1: Trade-Offs in Data Systems Architecture

#### *Systems of record*

A system of record, also known as a *source of truth*, holds the authoritative or *canonical* version of data. When new data comes in—for example, as user input—it is first written here. Each fact is represented exactly once (the representation is typically *normalized*; see “**Normalization, Denormalization, and Joins**” on page 72). If there is any discrepancy between another system and the system of record, the value in the system of record is (by definition) the correct one.

#### *Derived data systems*

Data in a derived system is the result of taking existing data from another system and transforming or processing it in some way. If you lose derived data, you can re-create it from the original source. A classic example is a cache: data can be served from the cache if present, but if the cache doesn’t contain what you need, you can fall back to the underlying database. Denormalized values, indexes, materialized views, transformed data representations, and models trained on a dataset also fall into this category.

Technically speaking, derived data is *redundant*, in the sense that it duplicates existing information. However, this data is often essential for getting good performance on read queries. You can derive several datasets from a single source, enabling you to look at the data from different points of view.

Analytical systems are usually derived data systems, because they are consumers of data created elsewhere. Operational services may contain a mixture of systems of record and derived data systems. The systems of record are the primary databases to which data is first written, whereas the derived data systems are the indexes and caches that speed up common read operations, especially for queries that the system of record cannot answer efficiently.

Most databases, storage engines, and query languages are not inherently systems of record or derived systems. A database is just a tool; how you use it is up to you. The distinction between a system of record and a derived data system depends not on the tool, but on the way you use it in your application. By being clear about which data is derived from which other data, you can bring clarity to an otherwise confusing system architecture.

When the data in one system is derived from the data in another, you need a process for updating the derived data when the original in the system of record changes. Unfortunately, many databases are designed based on the assumption that your application will always need to use only that one database, and they do not make it easy to integrate multiple systems in order to propagate such updates. In **Chapter 11** we will discuss data pipelines as an approach to *data integration*, which allows us to compose multiple data systems to achieve things that one system alone cannot do.

Operational Versus Analytical Systems | 11

That brings us to the end of our comparison of analytics and transaction processing. In the next section we will examine another trade-off that you might have already seen debated multiple times.
