drop table if exists recon_id;
create table recon_id (
    id          int unsigned not null primary key auto_increment,
    location    varchar(64) not null,
    ts          timestamp default current_timestamp );

drop table if exists recon_params;
create table recon_params (
	id          int unsigned not null primary key auto_increment,
    name        varchar(64) not null,
    value       tinyblob default null,
    iter        int not null default 0,
    ts          timestamp default current_timestamp,
    recon_id     int references recon_id(id) );




