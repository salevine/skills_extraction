"""
Stage 6: Build a skills ontology from assembled pipeline output.

Pure aggregation — no LLM calls. Groups verified skill mentions by
normalized name, computes frequency/confidence/distribution stats,
and writes the ontology as JSON + CSV.

Can run as the final pipeline stage or standalone via --ontology-only.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import write_json
from .preprocessing import extract_description_fields

logger = logging.getLogger(__name__)

_STRIP_SUFFIXES = re.compile(
    r"\s+(skills?|proficiency|expertise|knowledge|experience|abilities|ability)$",
    re.IGNORECASE,
)
_COLLAPSE_WS = re.compile(r"\s+")

# --- Skill alias system ---
# Static table: maps canonical_key variants to their preferred canonical form.
# Applied after basic canonicalization (lowercase, whitespace-collapsed, suffix-stripped).
_SKILL_ALIASES: dict[str, str] = {
    # Abbreviation → full form
    "js": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "ai": "artificial intelligence",
    "ci/cd": "continuous integration/continuous deployment",
    "cicd": "continuous integration/continuous deployment",
    "qa": "quality assurance",
    "gcp": "google cloud platform",
    "aws": "amazon web services",
    "ux": "user experience",
    "ui": "user interface",
    "swe": "software engineer",
    "sde": "software engineer",
    "devops": "development operations",
    "devsecops": "development security operations",
    "nosql": "no-sql",
    # Microsoft product aliases
    "ms excel": "excel",
    "microsoft excel": "excel",
    "ms word": "word",
    "microsoft word": "word",
    "ms office": "microsoft office",
    "ms office 365": "office 365",
    "microsoft office365": "office 365",
    "ms office365": "office 365",
    "ms outlook": "outlook",
    "microsoft outlook": "outlook",
    "ms powerpoint": "powerpoint",
    "microsoft powerpoint": "powerpoint",
    "ms visio": "visio",
    "microsoft visio": "visio",
    "ms project": "microsoft project",
    "ms teams": "microsoft teams",
    "ms azure": "azure",
    "microsoft azure": "azure",
    "ms power bi": "power bi",
    "microsoft power bi": "power bi",
    "ms power automate": "power automate",
    "microsoft power automate": "power automate",
    "ms power apps": "power apps",
    "microsoft power apps": "power apps",
    "ms powerapps": "power apps",
    "microsoft power platform": "power platform",
    "ms power platform": "power platform",
    "ms sharepoint": "sharepoint",
    "microsoft sharepoint": "sharepoint",
    "ms access": "access",
    "microsoft access": "access",
    "ms sql": "sql",
    "microsoft sql": "sql",
    "ms sql server": "sql server",
    "microsoft sql server": "sql server",
    "ms azure devops": "azure devops",
    "microsoft azure devops": "azure devops",
    "ms intune": "intune",
    "microsoft intune": "intune",
    "microsoft active directory": "active directory",
    "ms dynamics": "dynamics",
    "microsoft dynamics": "dynamics",
    "ms dynamics 365": "dynamics 365",
    "microsoft dynamics 365": "dynamics 365",
    "microsoft .net": ".net",
    "ms powershell": "powershell",
    "microsoft windows": "windows",
    "ms windows": "windows",
    "microsoft hyper-v": "hyper-v",
    "microsoft sccm": "sccm",
    "microsoft ssrs": "ssrs",
    "ms copilot": "copilot",
    "microsoft copilot": "copilot",
    "microsoft copilot studio": "copilot studio",
    "microsoft defender": "defender",
    "ms defender": "defender",
    "microsoft sentinel": "sentinel",
    "microsoft purview": "purview",
    "ms purview": "purview",
    "microsoft entra id": "entra id",
    "microsoft entra": "entra",
    "microsoft fabric": "fabric",
    "microsoft teams": "teams",
    "ms o365": "office 365",
    "microsoft o365": "office 365",
    "o365": "office 365",
    "microsoft m365": "m365",
    "ms m365": "m365",
    "ms visual studio": "visual studio",
    "microsoft visual studio": "visual studio",
    "microsoft t-sql": "t-sql",
    # Version/variant collapsing
    "python 3": "python",
    "python 3.x": "python",
    "python programming": "python",
    "python scripting": "python",
    "c# programming": "c#",
    "java programming": "java",
    "r programming": "r",
    "sql queries": "sql",
    "sql query language": "sql",
    "sql language": "sql",
    "advanced sql": "sql",
    "complex sql queries": "sql",
    "sql databases": "sql",
    "sql database": "sql",
    # Common qualifier-suffix collapses
    "agile development": "agile",
    "agile framework": "agile",
    "agile tools": "agile",
    "ai tools": "artificial intelligence",
    "ai technologies": "artificial intelligence",
    "ai technology": "artificial intelligence",
    "ai applications": "artificial intelligence",
    "ai development": "artificial intelligence",
    "ai platform": "artificial intelligence",
    "ai/ml": "artificial intelligence/machine learning",
    "ai/ml technologies": "artificial intelligence/machine learning",
    "ai/ml applications": "artificial intelligence/machine learning",
    "ai/ml tools": "artificial intelligence/machine learning",
    "automation tools": "automation",
    "automation technologies": "automation",
    "automation framework": "automation",
    "cloud technologies": "cloud computing",
    "cloud technology": "cloud computing",
    "cloud platform": "cloud computing",
    "cloud development": "cloud computing",
    "cloud-based technologies": "cloud computing",
    "cloud-based tools": "cloud computing",
    "big data technologies": "big data",
    "big data tools": "big data",
    "ci/cd tools": "continuous integration/continuous deployment",
    "bi tools": "business intelligence",
    "business intelligence tools": "business intelligence",
    "analytics tools": "analytics",
    "data visualization tools": "data visualization",
    "automated testing tools": "automated testing",
    "collaboration tools": "collaboration",
    "collaboration software": "collaboration",
    "configuration management tools": "configuration management",
    "angular framework": "angular",
    # Scan-discovered merges (2026-05-12)
    # Qualifier suffixes — only where stripping doesn't change meaning
    "machine learning tools": "machine learning",
    "artificial intelligence technologies": "artificial intelligence",
    "generative ai tools": "generative ai",
    "generative ai technologies": "generative ai",
    "generative ai applications": "generative ai",
    "genai tools": "genai",
    "genai technologies": "genai",
    "genai applications": "genai",
    "data analysis tools": "data analysis",
    "data analysis software": "data analysis",
    "data integration tools": "data integration",
    "data visualization software": "data visualization",
    "data analytics tools": "data analytics",
    "data governance tools": "data governance",
    "data modeling tools": "data modeling",
    "data management tools": "data management",
    "data warehousing technologies": "data warehousing",
    "data science tools": "data science",
    "project management tools": "project management",
    "project management software": "project management",
    "etl tools": "etl",
    "etl framework": "etl",
    "etl technologies": "etl",
    "etl development": "etl",
    "etl/elt tools": "etl/elt",
    "test automation tools": "test automation",
    "test automation framework": "test automation",
    "security tools": "security",
    "security technologies": "security",
    "cybersecurity tools": "cybersecurity",
    "containerization tools": "containerization",
    "containerization technologies": "containerization",
    "observability tools": "observability",
    "monitoring tools": "monitoring",
    "mlops tools": "mlops",
    "ml ops tools": "ml ops",
    "reporting tools": "reporting",
    "erp software": "erp",
    "erp platform": "erp",
    "crm software": "crm",
    "crm tools": "crm",
    "saas applications": "saas",
    "saas tools": "saas",
    "version control tools": "version control",
    "version control software": "version control",
    "debugging tools": "debugging",
    "itil framework": "itil",
    "safe framework": "safe",
    "scrum framework": "scrum",
    "react framework": "react",
    "spring framework": "spring",
    ".net framework": ".net",
    ".net technologies": ".net",
    ".net development": ".net",
    ".net applications": ".net",
    "django framework": "django",
    "jira software": "jira",
    "servicenow platform": "servicenow",
    "salesforce platform": "salesforce",
    "salesforce development": "salesforce",
    "scripting language": "scripting",
    "networking technologies": "networking",
    "virtualization technologies": "virtualization",
    "orchestration tools": "orchestration",
    "orchestration technologies": "orchestration",
    "siem tools": "siem",
    "iam tools": "iam",
    "rpa tools": "rpa",
    "itsm tools": "itsm",
    "grc tools": "grc",
    "api development": "api",
    "api technologies": "api",
    "vpn technologies": "vpn",
    "endpoint management tools": "endpoint management",
    "infrastructure automation tools": "infrastructure automation",
    "atlassian tools": "atlassian",
    # Microsoft/MS remaining
    "microsoft operating systems": "operating systems",
    "microsoft collaboration": "collaboration",
    "microsoft .net": ".net",
    "ms teams": "teams",
    "microsoft office 365": "office 365",
    "microsoft office applications": "microsoft office",
    "microsoft office products": "microsoft office",
    "microsoft office tools": "microsoft office",
    "microsoft office software": "microsoft office",
    "ms office products": "microsoft office",
    "ms office applications": "microsoft office",
    "ms office tools": "microsoft office",
    "microsoft windows server": "windows server",
    "microsoft azure ai": "azure ai",
    "microsoft azure services": "azure services",
    "ms azure government": "azure government",
    "microsoft dynamics crm": "dynamics crm",
    "microsoft azure entra id": "azure entra id",
    # Version collapses (only where version is irrelevant)
    "java 8": "java",
    "java 17+": "java",
    "windows 10": "windows",
    "windows 11": "windows",
    "itil 4": "itil",
    "google analytics 4": "google analytics",
    "oauth 2.0": "oauth",
    "sql server 2017": "sql server",
    # Google Cloud
    "google cloud platform": "google cloud",
    "google cloud tools": "google cloud",
    # --- auto-merge additions from run_20260520_210029 duplicate_candidates (qualifier/prefix, untagged) ---
    "a/b testing tools": "a/b testing",
    "accessible technologies": "accessible",
    "adk framework": "adk",
    "advanced analytical tools": "advanced analytical skills",
    "advanced analytics tools": "advanced analytics",
    "advertising technology": "advertising",
    "agent development": "agent skills",
    "agent framework": "agent skills",
    "agentic ai development": "agentic ai",
    "agentic ai tools": "agentic ai",
    "agentic development": "agentic",
    "agentic development tools": "agentic development",
    "agentic framework": "agentic",
    "agentic technology": "agentic",
    "agile delivery tools": "agile delivery",
    "agile project management software": "agile project management",
    "agile/scrum development": "agile/scrum",
    "ai agent development": "ai agent",
    "ai assisted development tools": "ai assisted development",
    "ai engineering tools": "ai engineering",
    "ai product development": "ai product",
    "ai-assisted coding tools": "ai-assisted coding",
    "ai-assisted software development tools": "ai-assisted software development",
    "ai-driven applications": "ai-driven",
    "ai-driven development": "ai-driven",
    "ai-driven technologies": "ai-driven",
    "ai-driven testing tools": "ai-driven testing",
    "ai-enabled applications": "ai-enabled",
    "ai-first applications": "ai-first",
    "ai-native applications": "ai-native",
    "ai-powered applications": "ai-powered",
    "ai-powered tools": "ai-powered",
    "alerting tools": "alerting",
    "analysis tools": "analysis",
    "android applications": "android",
    "android development": "android",
    "annotation tools": "annotation",
    "api tools": "api",
    "apm tools": "apm",
    "appian platform": "appian",
    "application programming": "application",
    "application security tools": "application security",
    "application software": "application",
    "asp technologies": "asp",
    "assembly language": "assembly",
    "azure administrator": "microsoft azure administrator",
    "azure cloud platform": "azure cloud",
    "azure cloud technologies": "azure cloud",
    "back-end development": "back-end",
    "backend language": "backend",
    "backend technologies": "backend",
    "backup technologies": "backup",
    "bi tool": "bi",
    "blockchain applications": "blockchain",
    "blockchain technologies": "blockchain",
    "blockchain technology": "blockchain",
    "build tools": "build",
    "building software": "building",
    "business development tools": "business development",
    "business language": "business",
    "c programming": "c",
    "c++ development": "c++",
    "c++ programming": "c++",
    "cad tools": "cad",
    "ci/cd pipeline development": "ci/cd pipeline",
    "cli tools": "cli",
    "clinical development": "clinical",
    "clinical software": "clinical",
    "cloud applications": "cloud",
    "cloud computing technologies": "cloud computing",
    "cloud native applications": "cloud native",
    "cloud software": "cloud",
    "cloud-based applications": "cloud-based",
    "cloud-based development": "cloud-based",
    "cloud-native application": "cloud-native",
    "cloud-native application development": "cloud-native application",
    "cloud-native applications": "cloud-native",
    "cloud-native development": "cloud-native",
    "cloud-native technologies": "cloud-native",
    "code analysis tools": "code analysis",
    "code optimization tools": "code optimization",
    "coding tools": "coding",
    "complex development": "complex",
    "compute technologies": "compute",
    "computer vision applications": "computer vision",
    "construction technology": "construction",
    "contact center applications": "contact center",
    "contact center technology": "contact center",
    "containerized development": "containerized",
    "continuous delivery technologies": "continuous delivery",
    "continuous integration tools": "continuous integration",
    "conversational ai platform": "conversational ai",
    "crm platform": "crm",
    "custom tools": "custom",
    "customer applications": "customer experience",
    "customer service software": "customer service",
    "customer success platform": "customer success",
    "customer support tools": "customer support",
    "cybersecurity technologies": "cybersecurity",
    "dashboarding tools": "dashboarding",
    "data analytics framework": "data analytics",
    "data center technologies": "data center",
    "data engineering tools": "data engineering",
    "data management technologies": "data management",
    "data migration tools": "data migration",
    "data pipeline tools": "data pipeline",
    "data platform development": "data platform",
    "data warehouse technologies": "data warehouse",
    "data-driven applications": "data-driven",
    "data-intensive applications": "data-intensive",
    "database applications": "database",
    "database development": "database",
    "database management software": "database management",
    "database programming": "database",
    "database technologies": "database",
    "databricks platform": "databricks",
    "dbms technology": "dbms",
    "deep learning framework": "deep learning",
    "defect tracking tools": "defect tracking",
    "deployment tools": "deployment",
    "desktop applications": "desktop",
    "developer platform": "developer",
    "developer productivity tools": "developer productivity",
    "developer tools": "developer",
    "development tools": "development",
    "digital technologies": "digital",
    "digital tools": "digital",
    "ecosystem": "microsoft ecosystem",
    "eda tools": "eda",
    "ediscovery tools": "ediscovery",
    "email security tools": "email security",
    "embedded firmware development": "embedded firmware",
    "embedded systems development": "embedded systems",
    "employee development": "employee experience",
    "employee tools": "employee experience",
    "enablement tools": "enablement",
    "encryption technologies": "encryption",
    "end-to-end development": "end-to-end",
    "endpoint security tools": "endpoint security",
    "engineering software": "engineering",
    "engineering tools": "engineering",
    "english language": "english",
    "enterprise ai platform": "enterprise ai",
    "enterprise ai tools": "enterprise ai",
    "enterprise architecture tools": "enterprise architecture",
    "enterprise data platform": "enterprise data",
    "enterprise infrastructure software": "enterprise infrastructure",
    "enterprise integration technologies": "enterprise integration",
    "enterprise security technologies": "enterprise security",
    "enterprise security tools": "enterprise security",
    "enterprise software applications": "enterprise software",
    "enterprise technologies": "enterprise",
    "enterprise tools": "enterprise",
    "entra id (azure ad)": "microsoft entra id (azure ad)",
    "epic applications": "epic",
    "esri software": "esri",
    "esri technology": "esri",
    "evaluation tools": "evaluation",
    "event-driven applications": "event-driven",
    "figma tool": "figma",
    "finance tools": "finance",
    "financial planning tools": "financial planning",
    "financial technology": "financial",
    "firewall technologies": "firewall",
    "firmware development": "firmware",
    "forecasting tools": "forecasting",
    "foundry": "microsoft foundry",
    "front end applications": "front end",
    "front end development": "front end",
    "front-end development": "front-end",
    "frontend development": "frontend",
    "frontend framework": "frontend",
    "frontend technologies": "frontend",
    "full stack applications": "full stack",
    "full stack development": "full stack",
    "functional software": "functional",
    "gis applications": "gis",
    "gpu programming": "gpu",
    "graph": "microsoft graph",
    "gtm tools": "gtm",
    "hands-on development": "hands-on",
    "healthcare software": "healthcare",
    "healthcare technology": "healthcare",
    "high-quality applications": "high-quality",
    "incident response tools": "incident response",
    "industrial software": "industrial",
    "industrial technology": "industrial",
    "industry 4.0 technologies": "industry 4.0",
    "information technologies": "information",
    "infrastructure as code (iac) tools": "infrastructure as code (iac)",
    "infrastructure as code tools": "infrastructure as code",
    "infrastructure technologies": "infrastructure",
    "infrastructure tools": "infrastructure",
    "infrastructure-as-code tools": "infrastructure-as-code",
    "innovative technologies": "innovative",
    "insurance software": "insurance",
    "integration applications": "integration",
    "integration development": "integration",
    "integration technologies": "integration",
    "integration tools": "integration",
    "ios development": "ios",
    "ipaas platform": "ipaas",
    "it service management tools": "it service management",
    "it strategy development": "it strategy",
    "java spring framework": "java spring",
    "java technologies": "java",
    "javascript development": "javascript",
    "javascript framework": "javascript",
    "kernel programming": "kernel",
    "knowledge base platform": "knowledge base",
    "kpi development": "kpi",
    "kpi framework": "kpi",
    "lead routing tools": "lead routing",
    "leadership development": "leadership",
    "learning technologies": "learning",
    "legal technology": "legal",
    "licensing application": "licensing",
    "linear programming": "linear",
    "llm applications": "llm",
    "logging tools": "logging",
    "low-code/no-code tools": "low-code/no-code",
    "machine learning applications": "machine learning",
    "management tools": "management",
    "manufacturing software": "manufacturing",
    "marketing automation tools": "marketing automation",
    "marketing technology": "marketing",
    "marketing tools": "marketing",
    "mcp tools": "mcp",
    "mdm tools": "mdm",
    "medical device development": "medical device expertise",
    "microcontroller programming": "microcontroller",
    "microservice development": "microservice",
    "microsoft .net": "net",
    "microsoft 365 applications": "microsoft 365",
    "microsoft 365 platform": "microsoft 365",
    "microsoft 365 tools": "microsoft 365",
    "microsoft ai capabilities": "ai capabilities",
    "microsoft applications": "applications",
    "microsoft azure ml": "azure ml",
    "microsoft azure solutions architect": "azure solutions architect",
    "microsoft certifications": "certifications",
    "microsoft cloud services": "cloud services",
    "microsoft d365": "d365",
    "microsoft developer tools": "developer tools",
    "microsoft ecosystem technologies": "microsoft ecosystem",
    "microsoft exchange": "exchange",
    "microsoft graph api": "graph api",
    "microsoft tools": "tools",
    "mitre att&ck framework": "mitre att&ck",
    "mobile application": "mobile",
    "mobile applications": "mobile",
    "mobile development": "mobile",
    "modeling and simulation tools": "modeling and simulation",
    "modeling tools": "modeling",
    "modern ai technologies": "modern ai",
    "monitoring/observability tools": "monitoring/observability",
    "ms copilot studio": "copilot studio",
    "ms dataverse": "dataverse",
    "ms office suite tools": "ms office suite",
    "ms project plan": "project plan",
    "net development": "net",
    "net framework": "net",
    "net platform": "net",
    "net technologies": "net",
    "network monitoring tools": "network monitoring",
    "network performance monitoring tools": "network performance monitoring",
    "network programming": "network",
    "network security technologies": "network security",
    "network troubleshooting tools": "network troubleshooting",
    "new business development": "new business",
    "nosql database development": "nosql database",
    "nosql database technologies": "nosql database",
    "object-oriented programming language": "object-oriented programming",
    "oci language": "oci",
    "open source software": "open-source",
    "open-source software": "open-source",
    "operating system software": "operating system",
    "operational software": "operational expertise",
    "oracle cloud applications": "oracle cloud",
    "oracle fusion applications": "oracle fusion",
    "organizational development": "organizational skills",
    "people development": "people skills",
    "performance monitoring tools": "performance monitoring",
    "performance optimization tools": "performance optimization",
    "performance testing tools": "performance testing",
    "personalization technologies": "personalization",
    "physics-based modeling software": "physics-based modeling",
    "pipeline development": "pipeline",
    "platform development": "platform",
    "plc programming": "plc",
    "policy application": "policy",
    "policy development": "policy",
    "portals": "microsoft portals",
    "power bi development": "power bi",
    "power platform tools": "power platform",
    "ppm software": "ppm",
    "ppm tools": "ppm",
    "predictive and generative ai technologies": "predictive and generative ai",
    "presentation development": "presentation skills",
    "process automation technologies": "process automation",
    "product analytics tools": "product analytics",
    "product management tools": "product management",
    "production applications": "production",
    "productivity tools": "productivity",
    "products": "microsoft products",
    "profiling tools": "profiling",
    "program management tools": "program management",
    "programming language": "programming",
    "programming software": "programming",
    "project": "microsoft project",
    "project tools": "project",
    "prototype development": "prototype",
    "prototyping tools": "prototyping",
    "python development": "python",
    "rdbms technologies": "rdbms",
    "real-time applications": "real-time",
    "real-time software": "real-time",
    "relational database applications": "relational database",
    "relational database technologies": "relational database",
    "replication technologies": "replication",
    "report development": "report",
    "requirements development": "requirements",
    "requirements management tools": "requirements management",
    "rest api development": "rest api",
    "restful api development": "restful api",
    "restful apis platform": "restful apis",
    "reverse engineering tools": "reverse engineering",
    "risk management framework": "risk management",
    "roadmap development": "roadmap",
    "rpa development": "rpa",
    "sales enablement tools": "sales enablement",
    "sales technology": "sales",
    "scada software": "scada",
    "scalable applications": "scalable",
    "scalable software": "scalable",
    "scalable technology": "scalable",
    "scaled agile framework": "scaled agile",
    "scm tools": "scm",
    "scratch programming": "scratch",
    "scripting / programming": "scripting",
    "search technologies": "search",
    "security software development": "security software",
    "self-service analytics tools": "self-service analytics",
    "self-service applications": "self-service",
    "self-service tools": "self-service",
    "semiconductor technologies": "semiconductor",
    "serverless framework": "serverless",
    "serverless technologies": "serverless",
    "serverless technology": "serverless",
    "servicenow development": "servicenow",
    "sharepoint development": "sharepoint",
    "sharepoint framework": "sharepoint",
    "sharepoint technologies": "sharepoint",
    "shell programming": "shell",
    "siem technologies": "siem",
    "simulation tools": "simulation",
    "slack platform": "slack",
    "software applications": "software",
    "software development tools": "software development",
    "software tools": "software",
    "spring boot framework": "spring boot",
    "sql programming": "sql",
    "sql tools": "sql",
    "sre tools": "sre",
    "statistical programming": "statistical",
    "statistical software": "statistical",
    "storage technologies": "storage",
    "streaming technologies": "streaming",
    "structural analysis software": "structural analysis",
    "support platform": "support",
    "supporting applications": "supporting",
    "system development": "system",
    "system management tools": "system management",
    "system programming": "system",
    "system software": "system",
    "systems language": "systems",
    "systems programming": "systems",
    "systems software": "systems",
    "technical development": "technical expertise",
    "technical platform": "technical expertise",
    "technical tools": "technical expertise",
    "technology stack": "microsoft technology stack",
    "technology tools": "technology",
    "test tools": "test experience",
    "ticketing software": "ticketing",
    "ticketing technology": "ticketing",
    "troubleshooting tools": "troubleshooting",
    "user story development": "user story",
    "validation tools": "validation",
    "value realization framework": "value realization",
    "video conferencing tools": "video conferencing",
    "visualization tools": "visualization",
    "vpn software": "vpn",
    "vulnerability management tools": "vulnerability management",
    "web services development": "web services",
    "wireless technologies": "wireless",
    "workflow applications": "workflow",
    "workflow automation tools": "workflow automation",
    "workflow orchestration tools": "workflow orchestration",
    "workflow tools": "workflow",

}

_GENERIC_SUFFIXES = re.compile(
    r"\s+(programming|language|framework|platform|tools?|software|"
    r"technologies|technology|development|applications?)$",
    re.IGNORECASE,
)


def _resolve_alias(canon: str) -> str:
    if canon in _SKILL_ALIASES:
        return _SKILL_ALIASES[canon]
    stripped = _GENERIC_SUFFIXES.sub("", canon)
    if stripped != canon and stripped in _SKILL_ALIASES:
        return _SKILL_ALIASES[stripped]
    return canon


def _canonicalize(name: str) -> str:
    s = name.strip()
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = _COLLAPSE_WS.sub(" ", s)
    s = _STRIP_SUFFIXES.sub("", s)
    s = s.strip(" .,;:-/")
    s = _resolve_alias(s)
    return s


def _pick_display_name(canon: str, variants: set) -> str:
    if not variants:
        return canon
    for v in sorted(variants):
        if v[0].isupper() and v.lower() == canon:
            return v
    return sorted(variants)[0]


def build_ontology(
    augmented_jobs: List[Dict[str, Any]],
    run_id: str,
    min_mention_count: int = 1,
) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "variants": set(),
        "type_counts": Counter(),
        "requirement_counts": Counter(),
        "confidences": [],
        "job_ids": set(),
        "job_titles": Counter(),
    })

    for job in augmented_jobs:
        job_id = job.get("id", "")
        title = extract_description_fields(job)[0]
        for m in job.get("skill_mentions") or []:
            if not m.get("is_skill"):
                continue
            raw_span = (m.get("skill_span") or "").strip()
            normalized = (m.get("normalized_candidate") or raw_span).strip()
            if not normalized:
                continue

            canon = _canonicalize(normalized)
            if not canon:
                continue

            g = groups[canon]
            g["variants"].add(normalized)
            if raw_span and raw_span != normalized:
                g["variants"].add(raw_span)
            g["type_counts"][m.get("hard_soft", "unknown")] += 1
            g["requirement_counts"][m.get("requirement_level", "unclear")] += 1
            conf = m.get("final_confidence")
            if conf is not None:
                g["confidences"].append(float(conf))
            g["job_ids"].add(job_id)
            if title:
                g["job_titles"][title] += 1

    ontology: List[Dict[str, Any]] = []
    for canon, g in groups.items():
        mention_count = sum(g["type_counts"].values())
        if mention_count < min_mention_count:
            continue

        confs = g["confidences"]
        avg_conf = round(sum(confs) / len(confs), 4) if confs else None
        conf_min = round(min(confs), 4) if confs else None
        conf_max = round(max(confs), 4) if confs else None

        display_name = _pick_display_name(canon, g["variants"])
        display_lower = display_name.lower()
        variants_list = sorted(v for v in g["variants"] if v.lower() != display_lower)

        ontology.append({
            "canonical_skill": display_name,
            "canonical_key": canon,
            "variants": variants_list,
            "type": g["type_counts"].most_common(1)[0][0] if g["type_counts"] else "unknown",
            "type_distribution": dict(g["type_counts"]),
            "requirement_level": g["requirement_counts"].most_common(1)[0][0] if g["requirement_counts"] else "unclear",
            "requirement_distribution": dict(g["requirement_counts"]),
            "job_count": len(g["job_ids"]),
            "mention_count": mention_count,
            "avg_confidence": avg_conf,
            "confidence_range": [conf_min, conf_max],
            "common_contexts": [t for t, _ in g["job_titles"].most_common(10)],
            "run_id": run_id,
        })

    ontology.sort(key=lambda x: (-x["job_count"], -x["mention_count"]))
    return ontology


def write_ontology_json(path: Path, ontology: List[Dict[str, Any]]) -> None:
    write_json(path, ontology)
    logger.info("Wrote ontology JSON (%d skills): %s", len(ontology), path)


def write_ontology_csv(path: Path, ontology: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "canonical_skill", "canonical_key", "variants", "type",
        "type_distribution", "requirement_level", "requirement_distribution",
        "job_count", "mention_count", "avg_confidence",
        "confidence_min", "confidence_max", "common_contexts", "run_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for entry in ontology:
            cr = entry.get("confidence_range") or [None, None]
            w.writerow({
                "canonical_skill": entry["canonical_skill"],
                "canonical_key": entry["canonical_key"],
                "variants": "; ".join(entry.get("variants", [])),
                "type": entry["type"],
                "type_distribution": json.dumps(entry.get("type_distribution", {})),
                "requirement_level": entry["requirement_level"],
                "requirement_distribution": json.dumps(entry.get("requirement_distribution", {})),
                "job_count": entry["job_count"],
                "mention_count": entry["mention_count"],
                "avg_confidence": entry.get("avg_confidence", ""),
                "confidence_min": cr[0] or "",
                "confidence_max": cr[1] or "",
                "common_contexts": "; ".join(entry.get("common_contexts", [])),
                "run_id": entry["run_id"],
            })
    logger.info("Wrote ontology CSV (%d skills): %s", len(ontology), path)


def build_ontology_from_file(
    augmented_path: Path,
    output_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    min_mention_count: int = 1,
) -> List[Dict[str, Any]]:
    augmented_path = Path(augmented_path)
    if not augmented_path.exists():
        raise FileNotFoundError(f"Augmented file not found: {augmented_path}")

    jobs = json.loads(augmented_path.read_text(encoding="utf-8"))
    if isinstance(jobs, dict):
        jobs = jobs.get("jobs", jobs.get("data", [jobs]))

    if run_id is None:
        m = re.search(r"run_(\S+?)\.json", augmented_path.name)
        run_id = m.group(1) if m else "unknown"

    if output_dir is None:
        output_dir = augmented_path.parent

    ontology = build_ontology(jobs, run_id, min_mention_count=min_mention_count)

    json_path = output_dir / f"SkillsExtraction_ontology_run_{run_id}.json"
    csv_path = output_dir / f"SkillsExtraction_ontology_run_{run_id}.csv"
    write_ontology_json(json_path, ontology)
    write_ontology_csv(csv_path, ontology)

    print(f"Ontology: {len(ontology)} canonical skills from {len(jobs)} jobs")
    print(f"  JSON: {json_path.name}")
    print(f"  CSV:  {csv_path.name}")

    return ontology


def find_potential_duplicates(
    ontology: List[Dict[str, Any]],
    min_jobs: int = 2,
) -> List[Dict[str, Any]]:
    """Detect potential duplicate skills in the ontology using non-LLM heuristics.

    Checks:
    1. Substring containment — shorter skill is a whole-word substring of longer
    2. Microsoft/MS prefix — "Microsoft X" or "MS X" where "X" also exists
    3. Qualifier suffixes — "X programming", "X tools", etc. where "X" exists
    4. Version variants — "X 3", "X 2.0" where "X" exists

    Returns a list of {pair, reason, suggestion} dicts for human review.
    """
    by_key = {e["canonical_key"]: e for e in ontology}
    keys_set = set(by_key.keys())
    results: List[Dict[str, Any]] = []
    seen_pairs: set = set()

    def _add(k1: str, k2: str, reason: str, target: str) -> None:
        pair = tuple(sorted([k1, k2]))
        if pair in seen_pairs:
            return
        seen_pairs.add(pair)
        e1, e2 = by_key[k1], by_key[k2]
        results.append({
            "skill_a": e1["canonical_skill"],
            "skill_a_jobs": e1["job_count"],
            "skill_b": e2["canonical_skill"],
            "skill_b_jobs": e2["job_count"],
            "reason": reason,
            "suggested_target": target,
        })

    for key in keys_set:
        entry = by_key[key]
        if entry["job_count"] < min_jobs:
            continue

        # Microsoft/MS prefix
        for prefix, label in [("microsoft ", "microsoft_prefix"), ("ms ", "ms_prefix")]:
            if key.startswith(prefix):
                base = key[len(prefix):]
                if base in keys_set:
                    target = base if by_key[base]["job_count"] >= entry["job_count"] else key
                    _add(key, base, label, by_key[target]["canonical_skill"])

        # Qualifier suffixes
        for suffix in [" programming", " language", " framework", " platform",
                       " tools", " tool", " software", " technologies",
                       " technology", " development", " applications", " application"]:
            if key.endswith(suffix):
                base = key[:-len(suffix)]
                if base in keys_set:
                    _add(key, base, "qualifier_suffix", by_key[base]["canonical_skill"])
                break

        # Version variants: strip trailing version numbers
        base = re.sub(r"\s+\d+(\.\d+)*(\.x)?$", "", key)
        if base != key and base in keys_set:
            _add(key, base, "version_variant", by_key[base]["canonical_skill"])

    # Substring containment (only among skills with min_jobs)
    significant = [k for k in keys_set if by_key[k]["job_count"] >= min_jobs and len(k) >= 3]
    significant.sort(key=lambda k: len(k))
    for i, short in enumerate(significant):
        if len(short) < 3:
            continue
        pattern = re.compile(r"\b" + re.escape(short) + r"\b")
        for long in significant[i + 1:]:
            if len(long) <= len(short):
                continue
            if pattern.search(long):
                short_jobs = by_key[short]["job_count"]
                long_jobs = by_key[long]["job_count"]
                if short_jobs >= 20 or long_jobs >= 20:
                    target = short if short_jobs >= long_jobs else long
                    _add(short, long, "substring_containment",
                         by_key[target]["canonical_skill"])

    results.sort(key=lambda x: -(x["skill_a_jobs"] + x["skill_b_jobs"]))
    return results


def write_duplicate_report(path: Path, duplicates: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(duplicates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote duplicate candidates (%d pairs): %s", len(duplicates), path)
