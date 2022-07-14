# Import required library
import tenseal as ts
import base64

context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree = 8192, coeff_mod_bit_sizes = [60, 40, 40, 60])
context.generate_galois_keys()
context.global_scale = 2**40

def write_data(file_name, data):
    if type(data) == bytes:
        #bytes to base64
        data = base64.b64encode(data)
         
    with open(file_name, 'wb') as f: 
        f.write(data)
 
def read_data(file_name):
    with open(file_name, "rb") as f:
        data = f.read()
    #base64 to bytes
    return base64.b64decode(data)

if __name__ == "__main__":
    secret_context = context.serialize(save_secret_key = True)
    write_data("./secret.txt", secret_context)
    
    context.make_context_public()
    public_context = context.serialize()
    write_data("./public.txt", public_context)
