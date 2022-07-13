import tenseal as ts
from authenticate import generate_keys as gk
import os, random

def encrypt_client_side(img_embedding, file_name):
    context = ts.context_from(gk.read_data("secret.txt"))
    enc_v = ts.ckks_vector(context, img_embedding)
    enc_v_proto = enc_v.serialize()
    gk.write_data(file_name, enc_v_proto)

def decrypt_client_side(file_name):
    context = ts.context_from(gk.read_data("secret.txt"))
    dec_v_proto = gk.read_data(file_name)
    dec_v = ts.lazy_ckks_vector_from(dec_v_proto)
    dec_v.link_context(context)
    return dec_v.decrypt()

def server_calculation(file_name):
    # True if verified with the profile else false
    context = ts.context_from(gk.read_data("public.txt"))

    enc_v1_proto = gk.read_data("profile.txt")
    enc_v1 = ts.lazy_ckks_vector_from(enc_v1_proto)
    enc_v1.link_context(context)

    enc_v2_proto = gk.read_data(file_name=file_name)
    enc_v2 = ts.lazy_ckks_vector_from(enc_v2_proto)
    enc_v2.link_context(context)

    euclidean_squared = enc_v1 - enc_v2
    euclidean_squared = euclidean_squared.dot(euclidean_squared)

    gk.write_data("result.txt", euclidean_squared.serialize())

def server_calculation_v1(test_path, base_path, result_path):
    context = ts.context_from(gk.read_data("public.txt"))
    for j, i in enumerate(os.listdir(test_path)):
        if i.endswith('.txt'):
            t1_proto = gk.read_data(test_path+i)
            t1 = ts.lazy_ckks_vector_from(t1_proto)
            t1.link_context(context)

            p1_proto = gk.read_data(base_path+random.choice([a for a in os.listdir(base_path) if a.endswith('.txt')]))
            p1 = ts.lazy_ckks_vector_from(p1_proto)
            p1.link_context(context)

            euclidean_squared = p1 - t1
            euclidean_squared = euclidean_squared.dot(euclidean_squared)

            gk.write_data(result_path+f"result{j}.txt", euclidean_squared.serialize())

def decrypt_client_side_v1(filepath):
    context = ts.context_from(gk.read_data("secret.txt"))
    for i in os.listdir(filepath):
        if i.endswith('.txt'):
            dec_v_proto = gk.read_data(filepath+i)
            dec_v = ts.lazy_ckks_vector_from(dec_v_proto)
            dec_v.link_context(context)
            print(dec_v.decrypt())
