CREATE TABLE faqs (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    question        VARCHAR(300) NOT NULL,
    answer          TEXT NOT NULL,
    category        VARCHAR(100) DEFAULT NULL,           -- optional: "admission", "fees", "hostel", etc.
    status          ENUM('active','inactive') DEFAULT 'active',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);