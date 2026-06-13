CREATE TABLE IF NOT EXISTS criminals (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    dob DATE NULL,
    moles TEXT NULL,
    nationality VARCHAR(100) NULL,
    region VARCHAR(100) NULL,
    crime TEXT NULL,
    num_crimes INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS criminal_images (
    id INT AUTO_INCREMENT PRIMARY KEY,
    criminal_id INT NOT NULL,
    image_path VARCHAR(500) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_criminal_images_criminal
        FOREIGN KEY (criminal_id) REFERENCES criminals(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS prediction_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    source_type VARCHAR(40) NOT NULL,
    source_ref VARCHAR(500) NULL,
    is_criminal TINYINT(1) NOT NULL,
    predicted_label INT NULL,
    confidence DOUBLE NOT NULL,
    status VARCHAR(100) NOT NULL,
    embedding_json LONGTEXT NOT NULL,
    selected_features_json LONGTEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
