CREATE TABLE docs01 (id SERIAL, doc TEXT, PRIMARY KEY(id));
INSERT INTO docs01 (doc) VALUES
('This is a good example of how Python and the Python language are acting'),
('as an intermediary between you the end user and me the programmer'),
('Python is a way for us to exchange useful instruction sequences ie'),
('programs in a common language that can be used by anyone who installs'),
('Python on their computer So neither of us are talking to'),
('Python instead we are communicating with each other'),
('The building blocks of programs'),
('In the next few chapters we will learn more about the vocabulary'),
('sentence structure paragraph structure and story structure of Python'),
('We will learn about the powerful capabilities of Python and how to');


CREATE TABLE invert01 (
  keyword TEXT,
  doc_id INTEGER REFERENCES docs02(id) ON DELETE CASCADE
);

INSERT INTO stop_words (word) VALUES 
('i'), ('a'), ('about'), ('an'), ('are'), ('as'), ('at'), ('be'), 
('by'), ('com'), ('for'), ('from'), ('how'), ('in'), ('is'), ('it'), ('of'), 
('on'), ('or'), ('that'), ('the'), ('this'), ('to'), ('was'), ('what'), 
('when'), ('where'), ('who'), ('will'), ('with');


insert into invert01(doc_id, keyword) 
select distinct id, lower(s.keyword) as keyword 
from docs01 as d, unnest(string_to_array(d.doc, ' ')) s(keyword) 
where lower(s.keyword) not in (select word from stop_words) order by id;
