# TO BE EXECUTED ONLY ONCE


DROP TABLE IF EXISTS executer;
DROP TABLE IF EXISTS scheduler;
DROP TABLE IF EXISTS users;

CREATE TABLE users(
    u_id int NOT NULL UNIQUE AUTO_INCREMENT,
    username varchar(255) NOT NULL,
    passwd varchar(255) NOT NULL,
    PRIMARY KEY (u_id)
);

CREATE TABLE scheduler(
    t_id int NOT NULL UNIQUE AUTO_INCREMENT,
    u_id INT NOT NULL,
    training_module varchar(255) UNIQUE,
    entry TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (t_id),
    FOREIGN KEY (u_id) REFERENCES users(u_id)
);

CREATE TABLE executer(
    e_id int NOT NULL UNIQUE AUTO_INCREMENT,
    t_id int NOT NULL,
    u_id int,
    training_module varchar(255),
    accuracy DECIMAL(10,8),
    kappa DECIMAL(10,8),
    entry TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (e_id),
    FOREIGN KEY (t_id) REFERENCES scheduler(t_id),
    FOREIGN KEY (training_module) REFERENCES scheduler(training_module),
    FOREIGN KEY (u_id) REFERENCES users(u_id)

);
