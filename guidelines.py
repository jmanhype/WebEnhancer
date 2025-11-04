"""Guidelines and entity definitions for information extraction.

This module defines the entity types and relations that the model should extract.
Customize these definitions based on your specific use case.
"""
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Relation:
    """Base class for relation annotations.

    A relation represents a connection between two or more entities in text.
    """
    subject: str
    predicate: str
    object: str

    def __repr__(self):
        return f"Relation(subject='{self.subject}', predicate='{self.predicate}', object='{self.object}')"


@dataclass
class Entity:
    """Base class for entity annotations.

    An entity represents a named element in text (person, organization, location, etc.)
    """
    text: str
    type: str
    start: Optional[int] = None
    end: Optional[int] = None

    def __repr__(self):
        return f"Entity(text='{self.text}', type='{self.type}')"


# Define valid entity types for your extraction task
ENTITY_DEFINITIONS = [Relation, Entity]


# Define extraction guidelines as strings that will be used in the prompt
guidelines = [
    "# Entity and Relation Extraction Guidelines",
    "",
    "## Entities",
    "Extract the following types of entities from the text:",
    "- PERSON: Names of people",
    "- ORGANIZATION: Companies, institutions, agencies",
    "- LOCATION: Cities, countries, landmarks",
    "- DATE: Temporal expressions",
    "- EVENT: Named events, incidents, occurrences",
    "",
    "## Relations",
    "Extract relationships between entities:",
    "- WORKS_FOR: Person works for Organization",
    "- LOCATED_IN: Entity is located in Location",
    "- PARTICIPATED_IN: Person or Organization participated in Event",
    "- OCCURRED_ON: Event occurred on Date",
    "",
    "## Instructions",
    "1. Identify all entities and their types",
    "2. Identify relationships between entities",
    "3. Return results as a list of Entity and Relation objects",
    "4. Only extract information explicitly stated in the text",
    "5. Do not infer or hallucinate information",
]
