from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Albert API
    albert_api_url: str = Field(
        default="https://albert.api.dev.etalab.gouv.fr",
        description="Base URL for the Albert API",
    )
    albert_api_token: str = Field(
        description="Bearer token for Albert API authentication",
    )

    # HuggingFace
    huggingface_token: str = Field(
        description="HuggingFace API token for dataset access",
    )

    # Database
    database_url: str = Field(
        description="SQLAlchemy database URL (e.g. postgresql://user:pass@host:5432/dbname?sslmode=require)",
    )

    # Sync settings
    chunk_batch_size: int = Field(
        default=64,
        description="Maximum number of chunks per API request",
    )
    requests_per_second: float = Field(
        default=2.0,
        description="Maximum chunk-upload requests per second (rate limiting)",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    # Tchap notifications (optional)
    tchap_homeserver: str = Field(
        default="https://matrix.agent.dinum.tchap.gouv.fr",
        description="Tchap Matrix homeserver URL",
    )
    tchap_access_token: str | None = Field(
        default=None,
        description="Tchap access token — notifications disabled if not set",
    )
    tchap_room_id: str | None = Field(
        default=None,
        description="Tchap room ID to send notifications to",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Mapping from HuggingFace dataset name to Albert API collection name and description
DATASET_COLLECTION_NAMES: dict[str, str] = {
    "AgentPublic/legi": "mediatech-legifrance",
    "AgentPublic/travail-emploi": "mediatech-fiches-travail-emploi",
    "AgentPublic/service-public": "mediatech-fiches-service-public",
    "AgentPublic/local-administrations-directory": "mediatech-annuaire-services-publics-locaux",
    "AgentPublic/state-administrations-directory": "mediatech-annuaire-services-publics-nationaux",
    "AgentPublic/dole": "mediatech-dossiers-legislatifs",
    "AgentPublic/constit": "mediatech-decisions-conseil-constitutionnel",
    "AgentPublic/cnil": "mediatech-decisions-cnil",
}

DATASET_COLLECTION_DESCRIPTIONS: dict[str, str] = {
    "AgentPublic/legi": (
        "L'intégral consolidé de la législation et de la réglementation nationale.\n"
        "Il est essentiellement constitué par :\n"
        "- les codes officiels ;\n"
        "- les lois, décrets-lois, ordonnances, décrets et une sélection d'arrêtés.\n"
        "- La consolidation des textes consiste lorsqu'un article de texte (ou de code) est modifié, à le réécrire en y intégrant la modification apportée. Les versions modifiées ou abrogées sont présentes dans le fonds documentaire au même titre que les versions en vigueur.\n"
        "- cette collection est exhaustive pour les lois, décrets-lois, ordonnances, décrets depuis 1945.\n"
        "Il y a en plus une sélection d'arrêtés consolidés et une autre sélection des autres arrêtés, en version originale, accessible dans la base JORF depuis 1990.\n"
        "Cette collection comprend les 73 codes officiels en vigueur consolidés (et les autres 29 abrogés).\n"
        "Seules certaines circulaires font l'objet d'une publication au Journal officiel ; pour l'essentiel, les circulaires ministérielles sont publiées dans le Bulletin officiel du ministère concerné (se reporter à la rubrique de leur site) ou sur le site Circulaires et instructions applicables (voir CIRCULAIRES sur data.gouv.fr).\n"
        "Source : https://www.legifrance.gouv.fr/"
    ),
    "AgentPublic/travail-emploi": (
        "Fiches pédagogiques du Ministère du travail sur les questions relatives au droit du travail.\n" "Source : https://travail-emploi.gouv.fr/"
    ),
    "AgentPublic/service-public": (
        'Fiches issues du site Service-Public.gouv.fr, site à destination des usagers. C\'est un service généraliste qui répond, le plus précisément possible, aux questions que se posent les particuliers ou les associations face aux situations ou difficultés les plus courantes, telles que : "Ai-je droit à une aide au logement ?", "Quel est le coût d\'un passeport ?", "Comment obtenir le formulaire de demande d\'aide juridictionnelle ?", "Puis-je faire mon changement d\'adresse en ligne ?".\n\n'
        "Le site contient des fiches pratiques d'informations et des ressources utiles (formulaires, démarches en ligne, textes de référence, sites web publics, etc.) pour connaître et comprendre ses droits et obligations et réaliser des démarches administratives.\n"
        "Source : https://www.service-public.gouv.fr/"
    ),
    "AgentPublic/local-administrations-directory": (
        "La Base de données locales référence plus de 86 000 guichets publics locaux (mairies, organismes sociaux, services de l'état, etc.). Elle fournit leurs coordonnées (adresses, téléphones, site internet, horaires d'ouverture, coordonnées de géolocalisation). En complément, sont indexés plus de 36 000 fichiers des communes (conformes au Code Officiel Géographique de l'INSEE), précisant la compétence géographique des guichets.\n"
        "Source : https://www.data.gouv.fr/datasets/service-public-gouv-fr-annuaire-de-ladministration-base-de-donnees-locales"
    ),
    "AgentPublic/state-administrations-directory": (
        "Le Référentiel de l'organisation administrative de l'Etat, nouvelle appellation de la base, comprend toutes les institutions régies par la Constitution de la Ve République et les administrations qui en dépendent, soit environ 6000 organismes.\n"
        "Le périmètre couvre les services centraux de l'Etat, jusqu'au niveau des bureaux.\n\n"
        "Le référentiel comprend les missions, l'organisation hiérarchique des services, leurs coordonnées complètes et le nom de leur(s) responsable(s).\n"
        "Source : https://www.data.gouv.fr/datasets/referentiel-de-lorganisation-administrative-de-leta"
    ),
    "AgentPublic/dole": (
        "Lois publiées depuis le début de la XIIe législature (juin 2002), les ordonnances publiées depuis 2002, et les lois en préparation (projets et propositions).\n"
        "Les dossiers législatifs permettent d'apporter des informations en amont et en aval de la promulgation des textes législatifs.\n"
        "Les dossiers législatifs portent sur les lois de l'article 39 de la Constitution. Lorsqu'il est décidé par une assemblée parlementaire de ne pas recourir à l'examen en forme simplifiée des textes relevant de l'article 53 de la Constitution, un dossier législatif est également ouvert.\n"
        "Depuis l'entrée en vigueur de la réforme constitutionnelle de 2008, les dossiers législatifs de propositions de loi ne sont ouverts qu'après l'adoption du texte par la première assemblée saisie.\n"
        "Source : https://www.data.gouv.fr/datasets/dole-les-dossiers-legislatifs"
    ),
    "AgentPublic/constit": (
        "Références de toutes les décisions du Conseil constitutionnel depuis sa création en 1958 et ces mêmes décisions en texte intégral selon le tableau suivant :\n\n"
        "** Contentieux des normes **\n"
        "Décisions constitutionnelles (DC) depuis l'origine (1958), Question prioritaire de constitutionnalité (QPC) depuis l'origine (2010), Contrôle des lois (LP) du pays (Nouvelle calédonie et Polynésie française) depuis l'origine (1958), Contrôle des lois d'outre mer (LOM) depuis l'origine (2007), les déclassements de textes (L) depuis l'origine (1958), les fins de non recevoir (FNR) depuis l'origine (1958),\n\n"
        "**Contentieux électoral et assimilé **\n"
        "AN depuis 1993, Sénat depuis 1993, Présidentielle depuis 1993, Référendum depuis 1993, Déchéance (D) depuis 1985, Incompatibilités (I) depuis l'origine (1958)\n\n"
        "**Autres **\n"
        "Nominations (membres, rapporteurs adjoints, secrétaires généraux), organisation, autres décisions ... depuis 1997\n"
        "Article 16 depuis l'origine (1958)\n\n"
        "L'ensemble des décisions du Conseil constitutionnel est publié au Journal officiel de la République française ainsi qu'au recueil annuel des décisions du Conseil constitutionnel.\n"
        "Source : https://www.data.gouv.fr/datasets/constit-les-decisions-du-conseil-constitutionnel"
    ),
    "AgentPublic/cnil": (
        "Toutes les délibérations de la CNIL depuis l'origine, la première délibération ayant été rendue en 1979.\n"
        "Leurs modalités de publication sont définies par la loi Informatique et Libertés, par son décret d'application ainsi que par le règlement intérieur de la Commission.\n\n"
        "Les délibérations adoptées entre le 1er janvier 1979 et le 3 mai 2012 ont été publiées dans leur version mise à jour au plus tard le 4 mai 2012. Les délibérations adoptées postérieurement au 3 mai 2012 portent mention de leur date de publication effective.\n"
        "Source : https://echanges.dila.gouv.fr/OPENDATA/CNIL/"
    ),
}

# Datasets to sync
DATASETS = [
    "AgentPublic/legi",
    "AgentPublic/travail-emploi",
    "AgentPublic/service-public",
    "AgentPublic/local-administrations-directory",
    "AgentPublic/state-administrations-directory",
    "AgentPublic/dole",
    "AgentPublic/constit",
    "AgentPublic/cnil",
]

# Field used as document name for each dataset
DATASET_TITLE_FIELD: dict[str, str] = {
    "AgentPublic/legi": "full_title",
    "AgentPublic/travail-emploi": "title",
    "AgentPublic/service-public": "title",
    "AgentPublic/dole": "title",
    "AgentPublic/constit": "title",
    "AgentPublic/cnil": "title",
    "AgentPublic/local-administrations-directory": "name",
    "AgentPublic/state-administrations-directory": "name",
}

# Metadata fields to extract from each dataset
DATASET_METADATA_FIELDS: dict[str, list[str]] = {
    "AgentPublic/legi": [
        "category",
        "ministry",
        "status",
        "start_date",
        "end_date",
        "number",
    ],
    "AgentPublic/service-public": [
        "title",
        "audience",
        "theme",
        "url",
    ],
    "AgentPublic/travail-emploi": [
        "title",
        "date" "url",
    ],
    "AgentPublic/local-administrations-directory": [
        "name",
        "types",
        "siret",
        "siren",
        "mails",
        "modification_date",
        "directory_url",
    ],
    "AgentPublic/state-administrations-directory": [
        "name",
        "types",
        "siret",
        "siren",
        "mails",
        "modification_date",
        "directory_url",
    ],
    "AgentPublic/dole": [
        "title",
        "category",
        "number",
        "article_number",
        "creation_date",
    ],
    "AgentPublic/constit": [
        "title",
        "nature",
        "solution",
        "number",
        "decision_date",
    ],
    "AgentPublic/cnil": ["title", "nature", "nature_delib", "status", "number", "date"],
}


def get_settings() -> Settings:
    return Settings()
