<?php
    require_once("./mesas.php");
    //0900601232X
    @mkdir(__DIR__.'/images', 0777, true);
    chmod(__DIR__.'/images', 0777);

    foreach ($mesas as $mesa) {
        echo 'descargando: '.$mesa.'';
        touch(__DIR__.'/images/'.$mesa.'.jpg');
        $base64Data = descargar($mesa);
        $image = base64_to_jpeg($base64Data, __DIR__.'/images/'.$mesa.'.jpg' );
    }
    echo 'fin';
    exit;

    function descargar($mesa) {
        $endpoint = "https://resultados.gob.ar/backend-difu/scope/data/getTiff/$mesa";
        $curl = curl_init();

        curl_setopt_array($curl, array(
            CURLOPT_URL => 'https://resultados.gob.ar/backend-difu/scope/data/getTiff/'.$mesa,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_ENCODING => '',
            CURLOPT_MAXREDIRS => 10,
            CURLOPT_TIMEOUT => 0,
            CURLOPT_HTTPHEADER => array(
                'Accept: */*',
                'Connection: keep-alive',
                'Accept-Encoding: gzip, deflate, br',
                'User-Agent: PostmanRuntime/7.33.0',
              ),
            CURLOPT_HTTP_VERSION => CURL_HTTP_VERSION_1_1,
            CURLOPT_CUSTOMREQUEST => 'GET',
        ));

        $response = curl_exec($curl);
        $responseParsed = json_decode($response);    
        curl_close($curl);
        return $responseParsed->encodingBinary;
    }

    function base64_to_jpeg( $base64_string, $output_file ) {
        $status = file_put_contents($output_file, base64_decode($base64_string));
        chmod($output_file,777);
        return($status); 
    }
    
    
?>