<?php
    require_once("./DownloadImage.php");
    $directory = sanitizeInput($argv[1]);
    $mesa = sanitizeInput($argv[2]);
    $file = $directory.$mesa.'.jpg';
    run($file, $mesa);
    exit(0);

    function sanitizeInput($input) {
        return htmlspecialchars($input, ENT_QUOTES, 'UTF-8');
    }

    function base64_to_jpeg($base64_string, $output_file) {
        $status = file_put_contents($output_file, base64_decode($base64_string));
        chmod($output_file,0644);
        return $status; 
    }

    function run ($file, $mesa) {
        touch($file);
        $base64Data = DownloadImage::download($mesa);
        $resp = 1;
        
        if(empty($base64Data)) {
            $resp = 0;
        }
        
        if($resp) {
            $status = base64_to_jpeg($base64Data, $file);
            if(!$status) {
                $resp = 0;
            }
        }
        
        return $resp;
    }