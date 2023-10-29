<?php

class DownloadImage {
    public static function download($mesa) {
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
        curl_close($curl);
        return self::processJson($response, $mesa);
    }

    private static function processJson($response,  $mesa) {
        
        if($response == "NOT FOUND") {
            return false;
        }
        $responseParsed = json_decode($response);
        
        if(empty($responseParsed->encodingBinary)) {
            error_log('Telegrama faltante para la mesa: '.$mesa.'');
            file_put_contents(__DIR__."/error_mesa.txt", 'Telegrama faltante para la mesa '.$mesa.PHP_EOL, FILE_APPEND);
            return false;
        }
        return $responseParsed->encodingBinary;
    }
}
?>