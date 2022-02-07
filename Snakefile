

SESSION, CAMERA, VIDEO_NAMES, = glob_wildcards("/data/raw/{session}/cam{camera}/{rawmov}.MOV")


rule all:
    input:
        "data/final/movie_paths.csv"


rule extract_metadata:
    input:
        "/data/raw/{session}/cam{camera}/{rawmov}.MOV"
    output:
        "data/processed/{session}_cam-{camera}_mov-{rawmov}.json"
    conda:
        "environment.yml"
    shell:
        "python scripts/extract_metadata.py --camera {wildcards.camera} --session {wildcards.session} {input} > {output}"



rule merge_metadata_to_csv:
    input:
        expand("data/processed/{session}_cam-{camera}_mov-{rawmov}.json", zip, session=SESSION, camera=CAMERA, rawmov=VIDEO_NAMES)
    output:
        "data/final/movie_paths.csv"
    conda:
        "environment.yml"
    script:
        "scripts/merge_json_to_csv.py"
    