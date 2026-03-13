

# Retrouver id du premier document
curl -s "https://albert.api.dev.etalab.gouv.fr/v1/collections?name=travail-emploi" \
    -H "Authorization: Bearer $ALBERT_API_TOKEN" | jq '.data[0].id'
# 4

# Retrouver contenu du doc retrouvé ci dessus
curl -s "https://albert.api.dev.etalab.gouv.fr/v1/documents/{document_id}/chunks?limit=3" \
    -H "Authorization: Bearer $ALBERT_API_TOKEN" | jq .

# Supprimer collection
curl -s -X DELETE "https://albert.api.dev.etalab.gouv.fr/v1/collections/3" \
    -H "Authorization: Bearer $ALBERT_API_TOKEN"