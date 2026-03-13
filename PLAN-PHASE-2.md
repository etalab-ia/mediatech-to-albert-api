# Plan Phase 2

## Tâches

### Performance
- [ ] **Bottleneck à confirmer** : 95s pour 30Mo (travail-emploi) → ~9h pour legi (~2.14M rows). Probable bottleneck = batch_size=64 de l'API Albert. Optimiser côté API (augmenter la limite de batch), pas côté script.
- [ ] Mesurer temps par opération (create_document vs create_chunks) pour confirmer

### À vérifier
- [ ] **Limite 255 chars de l'API Albert** pour les noms de documents et les valeurs de métadonnées. Pour l'instant on tronque avec "..." — si l'API n'a pas cette limite, supprimer la troncature.

## Comportement si state.db est perdu

Si `state.db` est perdu (conteneur tué sans volume) : le script détecte l'absence de `albert_collection_id` local, trouve la collection existante dans Albert, la **supprime et recrée** pour éviter les doublons, puis re-uploade tous les documents. Correct mais coûteux. Monter `state.db` dans un volume Docker si le re-upload complet est trop coûteux.
