// -----------------------------------------------------------------------------
// Initial Knowledge Graph Schema, Constraints, and Vector Indexes
// -----------------------------------------------------------------------------

// 1. Constraints
CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT image_id_unique IF NOT EXISTS
FOR (i:Image) REQUIRE i.id IS UNIQUE;

CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.name IS UNIQUE;

CREATE CONSTRAINT community_id_unique IF NOT EXISTS
FOR (cm:Community) REQUIRE cm.id IS UNIQUE;

// 2. Lookup Indexes
CREATE INDEX entity_type_index IF NOT EXISTS
FOR (e:Entity) ON (e.entity_type);

CREATE INDEX chunk_doc_index IF NOT EXISTS
FOR (c:Chunk) ON (c.document_id);

CREATE INDEX image_asset_index IF NOT EXISTS
FOR (i:Image) ON (i.media_asset_id);
