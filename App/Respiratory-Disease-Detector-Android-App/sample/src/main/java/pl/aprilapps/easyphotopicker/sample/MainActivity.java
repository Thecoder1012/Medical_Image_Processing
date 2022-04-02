package pl.aprilapps.easyphotopicker.sample;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import pl.aprilapps.easyphotopicker.ChooserType;
import pl.aprilapps.easyphotopicker.DefaultCallback;
import pl.aprilapps.easyphotopicker.RespiratoryDisease;
import pl.aprilapps.easyphotopicker.MediaFile;
import pl.aprilapps.easyphotopicker.MediaSource;

import android.content.Context;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import org.pytorch.Module;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.IValue;
import org.pytorch.Tensor;
import org.pytorch.LiteModuleLoader;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity implements RespiratoryDisease.RespiratoryDiseaseStateHandler {

    private static final String TAG = MainActivity.class.getName();
    private static final String PHOTOS_KEY = "respiratory_disease_photos_list";
    private static final String STATE_KEY = "respiratory_disease_state";
    private static final int CHOOSER_PERMISSIONS_REQUEST_CODE = 7459;
    private static final int GALLERY_REQUEST_CODE = 7502;
    private static final int DOCUMENTS_REQUEST_CODE = 7503;
    private static final int LEGACY_EXTERNAL_STORAGE_PERMISSION_REQUEST_CODE = 456;

    protected RecyclerView recyclerView;

    protected View galleryButton;
    protected TextView textView;

    private ImagesAdapter imagesAdapter;

    private ArrayList<MediaFile> photos = new ArrayList<>();

    private RespiratoryDisease RespiratoryDisease;

    private static final String[] LEGACY_WRITE_PERMISSIONS = new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE};

    public static final int MY_PERMISSIONS_REQUEST_CAMERA = 100;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        recyclerView = findViewById(R.id.recycler_view);
        galleryButton = findViewById(R.id.gallery_button);
        textView = findViewById(R.id.pic_label);
        textView.setTextSize(25);
        textView.setTypeface(null, Typeface.BOLD);

        if (savedInstanceState != null) {
            photos = savedInstanceState.getParcelableArrayList(PHOTOS_KEY);
            RespiratoryDiseaseState = savedInstanceState.getParcelable(STATE_KEY);
        }

        imagesAdapter = new ImagesAdapter(this, photos);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setHasFixedSize(true);
        recyclerView.setAdapter(imagesAdapter);

        RespiratoryDisease = new RespiratoryDisease.Builder(this)
                .setChooserTitle("Pick media")
                .setCopyImagesToPublicGalleryFolder(true) // THIS requires granting WRITE_EXTERNAL_STORAGE permission for devices running Android 9 or lower
//                .setChooserType(ChooserType.CAMERA_AND_DOCUMENTS)
                .setChooserType(ChooserType.CAMERA_AND_GALLERY)
                .setFolderName(".")
                .allowMultiple(false)
                .setStateHandler(this)
                .build();


        findViewById(R.id.gallery_button).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (isLegacyExternalStoragePermissionRequired()) {
                    requestLegacyWriteExternalStoragePermission();
                } else {
                    RespiratoryDisease.openGallery(MainActivity.this);
                }
            }
        });


        findViewById(R.id.camera_button).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //Intent intent = new Intent("android.media.action.IMAGE_CAPTURE");
                //startActivity(intent);
                RespiratoryDisease.openCameraForImage(MainActivity.this);
//                if (ContextCompat.checkSelfPermission(MainActivity.this.getApplicationContext(),
//                        Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
//                    //Intent intent = new Intent("android.media.action.IMAGE_CAPTURE");
//                    //startActivity(intent);
//                    //RespiratoryDisease.openCameraForImage(MainActivity.this);
//                }else {
//                    ActivityCompat.requestPermissions(MainActivity.this,
//                            new String[]{Manifest.permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA);
//                }
            }

        });


        findViewById(R.id.documents_button).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (isLegacyExternalStoragePermissionRequired()) {
                    requestLegacyWriteExternalStoragePermission();
                } else {
                    //RespiratoryDisease.openDocuments(MainActivity.this);
                    try {
                        Module module = LiteModuleLoader.load(assetFilePath(MainActivity.this.getApplicationContext(), "DiseaseDetector.ptl"));
                        MediaFile mf = imagesAdapter.getMediaFile();
                        Bitmap bm = BitmapFactory.decodeFile(mf.getFile().getAbsolutePath());

                        Tensor image = TensorImageUtils.bitmapToFloat32Tensor(bm,
                                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

                        // Runs model inference and gets result.
                        Tensor outputTensor = module.forward(IValue.from(image)).toTensor();
                        float[] scores = outputTensor.getDataAsFloatArray();
                        for(int i = 0; i < scores.length; i++)
                            System.out.println(scores[i]);

                        float maxScore = -Float.MAX_VALUE;
                        int maxScoreIdx = -1;
                        for (int i = 0; i < scores.length; i++) {
                            if (scores[i] > maxScore) {
                                maxScore = scores[i];
                                maxScoreIdx = i;
                            }
                        }
                        String resultStr;
                        if(maxScoreIdx == 0) resultStr = "COVID-19" + ": " + maxScore;
                        else if(maxScoreIdx == 1) resultStr = "Lung Opacity" + ": " + maxScore;
                        else if(maxScoreIdx == 2) resultStr = "Normal" + ": " + maxScore;
                        else if(maxScoreIdx == 3) resultStr = "Pneumonia" + ": " + maxScore;
                        else resultStr = "Tuberculosis" + ": " + maxScore;

                        System.out.println(resultStr);

                        textView.setText(resultStr);

                        // Releases model resources if no longer used.
                        //model.close();
                    } catch (Exception e) {

                        Toast.makeText(getApplicationContext(),"Some Error Occurred! "+e.getMessage().toString(),Toast.LENGTH_SHORT).show();
                        // TODO Handle the exception
                        finish();
                    }
                }
            }
        });

        /*
        findViewById(R.id.chooser_button).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (isLegacyExternalStoragePermissionRequired()) {
                    requestLegacyWriteExternalStoragePermission();
                } else {
                    RespiratoryDisease.openChooser(MainActivity.this);
                }
            }
        });*/

    }

    private Bundle RespiratoryDiseaseState = new Bundle();

    @Override
    @NonNull
    public Bundle restoreRespiratoryDiseaseState() {
        return RespiratoryDiseaseState;
    }

    @Override
    public void saveRespiratoryDiseaseState(Bundle state) {
        RespiratoryDiseaseState = state;
    }

    @Override
    public void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        outState.putSerializable(PHOTOS_KEY, photos);
        outState.putParcelable(STATE_KEY, RespiratoryDiseaseState);
    }

    private void checkGalleryAppAvailability() {
        if (!RespiratoryDisease.canDeviceHandleGallery()) {
            //Device has no app that handles gallery intent
            galleryButton.setVisibility(View.GONE);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == CHOOSER_PERMISSIONS_REQUEST_CODE && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            RespiratoryDisease.openChooser(MainActivity.this);
        } else if (requestCode == GALLERY_REQUEST_CODE && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            RespiratoryDisease.openGallery(MainActivity.this);
        } else if (requestCode == DOCUMENTS_REQUEST_CODE && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            RespiratoryDisease.openDocuments(MainActivity.this);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        RespiratoryDisease.handleActivityResult(requestCode, resultCode, data, this, new DefaultCallback() {
            @Override
            public void onMediaFilesPicked(MediaFile[] imageFiles, MediaSource source) {
                for (MediaFile imageFile : imageFiles) {
                    Log.d("RespiratoryDisease", "Image file returned: " + imageFile.getFile().toString());
                }
                onPhotosReturned(imageFiles);
            }

            @Override
            public void onImagePickerError(@NonNull Throwable error, @NonNull MediaSource source) {
                //Some error handling
                error.printStackTrace();
            }

            @Override
            public void onCanceled(@NonNull MediaSource source) {
                //Not necessary to remove any files manually anymore
            }
        });
    }

    private void onPhotosReturned(@NonNull MediaFile[] returnedPhotos) {
        photos.addAll(Arrays.asList(returnedPhotos));
        imagesAdapter.notifyDataSetChanged();
        recyclerView.scrollToPosition(photos.size() - 1);
    }

    private boolean isLegacyExternalStoragePermissionRequired() {
        boolean permissionGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
        return Build.VERSION.SDK_INT < 29 && !permissionGranted;
    }

    private void requestLegacyWriteExternalStoragePermission() {
        ActivityCompat.requestPermissions(MainActivity.this, LEGACY_WRITE_PERMISSIONS, LEGACY_EXTERNAL_STORAGE_PERMISSION_REQUEST_CODE);
    }

    private String assetFilePath(Context context, String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        } catch (IOException e) {
            Log.e(TAG, assetName + ": " + e.getLocalizedMessage());
        }
        return null;
    }
}
