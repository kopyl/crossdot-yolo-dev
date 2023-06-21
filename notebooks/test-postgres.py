import psycopg2
import datetime
import time

db_init_args = {
    "database": "postgres",
    "host": "crossdot-postgres-2.c7noywaluiwu.us-east-1.rds.amazonaws.com",
    "user": "crossdot_db_user",
    "password": "vwWUQeXs37IzQlDTbIBN3KXxNZE7HQhT",
    "port": 5432,
}

# start_time = time.time()
conn = psycopg2.connect(**db_init_args)
cursor = conn.cursor()
# cursor.execute("select relname from pg_class where relkind='r' and relname !~ '^(pg_|sql_)';")
# print(cursor.fetchall())
# cursor.close()



# get all rows from users table
# cursor.execute("select * from users;")
# for user in cursor.fetchall():
#     print(user)
# cursor.close()

token = "a72a844e96cb116649a0bded89322592"

def get_user_by_token(token):
    cursor.execute("select * from api_tokens where token = %s;", (token,))
    return cursor.fetchone()


user = get_user_by_token(token)





# create_table_query = """
#     CREATE TABLE generated_images (
#         id SERIAL PRIMARY KEY,
#         created_at TIMESTAMP NOT NULL,
#         image_url VARCHAR(255) NOT NULL
#     );
# """
# cursor.execute(create_table_query)
# conn.commit()
# cursor.close()




# cursor.execute("select * from generated_images;")
# for image in cursor.fetchall():
#     print(image)
# cursor.close()



# created_at = datetime.datetime.now()
# image_url = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"
# cursor.execute("insert into generated_images (created_at, image_url) values (%s, %s);", (created_at, image_url))
# conn.commit()
# cursor.close()

# end_time = time.time()
# print("Time: ", end_time - start_time)