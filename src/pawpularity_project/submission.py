Path = ''
sample_submission = pd.read_csv('/Users/wmeikle/Downloads/petfinder-pawpularity-score/sample_submission.csv')
sample_submission['Id'] = sample_submission['Id'].apply(lambda x: '../input/petfinder-pawpularity-score/test/'+x+'.jpg')
sample_submission.to_csv('sample_submission.csv', index=False, header=False)
sample_submission = tf.data.TextLineDataset(
    'sample_submission.csv'
).map(decode_csv).batch(BATCH_SIZE)

sample_prediction = model.predict(sample_submission)

submission_output = pd.concat(
    [pd.read_csv('../input/petfinder-pawpularity-score/sample_submission.csv').drop('Pawpularity', axis=1),
    pd.DataFrame(sample_prediction)],
    axis=1
)
submission_output.columns = [['Id', 'Pawpularity']]

submission_output.to_csv('submission.csv', index=False)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/model.joblib")
    ap.add_argument("--input", required=True, help="CSV with feature columns")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--id-col", default="id", help="ID column for submission output")
    return ap.parse_args()


def main():
    args = parse_args()

    model = load(args.model)
    X = load_csv(args.input)

    out = model.predict(X)

    save_csv(out, args.output)
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
