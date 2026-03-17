

# Retrouver id du premier document
curl -s "https://albert.api.dev.etalab.gouv.fr/v1/collections?name=travail-emploi" \
    -H "Authorization: Bearer $ALBERT_API_TOKEN" | jq '.data[0].id'
# 5

# Retrouver contenu du doc retrouvé ci dessus
curl -s "https://albert.api.dev.etalab.gouv.fr/v1/documents/{document_id}/chunks?limit=3" \
    -H "Authorization: Bearer $ALBERT_API_TOKEN" | jq .

# Supprimer collection
curl -s -X DELETE "https://albert.api.dev.etalab.gouv.fr/v1/collections/5" \
    -H "Authorization: Bearer $ALBERT_API_TOKEN"


python main.py --dataset AgentPublic/service-public

python main.py --dataset AgentPublic/local-administrations-directory && \
python main.py --dataset AgentPublic/state-administrations-directory && \
python main.py --dataset AgentPublic/dole && \
python main.py --dataset AgentPublic/constit && \
python main.py --dataset AgentPublic/cnil && python main.py --status