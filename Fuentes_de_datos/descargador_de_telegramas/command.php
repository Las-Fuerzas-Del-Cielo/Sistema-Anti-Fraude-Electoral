<?php
    require_once("./DownloadImage.php");
    $directory = $argv[1];
    $mesa = $argv[2];
    $file = $directory.$mesa.'.jpg';
    run($file, $mesa);
    exit(0);


    function base64_to_jpeg($base64_string, $output_file) {
        $status = file_put_contents($output_file, base64_decode($base64_string));
        chmod($output_file,777);
        return $status; 
    }

    function run ($file, $mesa) {
        touch($file);
        $base64Data = DownloadImage::download($mesa);
        
        if(empty($base64Data)) {
            return 0;
        }
        
        $status = base64_to_jpeg($base64Data, $file);
        
        if(!$status) {
            return 0;
        }
        
        return 1;
    }